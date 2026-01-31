import torch
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from typing import Sequence, List, cast
from syntheseus.reaction_prediction.models.smiles_medusa.model import SmilesToSmilesAutoregressiveMedusaModel
from syntheseus.reaction_prediction.models.smiles_medusa.tokenizer import InplaceSMILESTokenizer

class SmilesMedusaModel(ExternalBackwardReactionModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        chkpt_path = get_unique_file_in_dir(self.model_dir, pattern="*.ckpt")
        vocab_path = get_unique_file_in_dir(self.model_dir, pattern="*.json")

        self.tokenizer = InplaceSMILESTokenizer()
        self.tokenizer.load_vocab(vocab_path)

        self.model = SmilesToSmilesAutoregressiveMedusaModel(
            medusa_heads_number=20,
            medusa_layers_number=1,
            medusa_hidden_dim=50,
            embedding_dim=256,
            feedforward_dim=2048,
            num_encoder_layers=6,
            num_decoder_layers=6,
            transformer_heads_number=8,
            dropout_rate=0.1,
            activation="gelu",
            learning_rate=3e-4,
            weight_decay=0.0,
            scheduler="const",
            warmup_steps=0,
            tokenizer=self.tokenizer,
            beam_size=10,
            max_size=200,
        )
        checkpoint = torch.load(chkpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_parameters(self):
        return self.model.parameters()

    def _mols_to_batch(self, inputs: List[Molecule]) -> torch.Tensor:
        smiles = [mol.smiles for mol in cast(List[Molecule], inputs)]
        return torch.tensor([self.tokenizer.encode(smi) for smi in smiles], dtype=torch.long, device=self.device)

    def _get_reactions(self, inputs: list[Molecule], num_results: int) -> list[Sequence[SingleProductReaction]]:
        src = self._mols_to_batch(inputs)
        output_batch, output_batch_probs = self.model.generate(src)
        output_batch_probs = output_batch_probs.detach().cpu().tolist()
        output_smiles = [self.tokenizer.decode_batch(b) for b in output_batch.cpu().tolist()]
        metadata = []
        for lst in output_batch_probs:
            metadata.append([{"probability": prob} for prob in lst])
        return [process_raw_smiles_outputs_backwards(inp, output, metadata) for inp, output, metadata in zip(inputs, output_smiles, metadata)]