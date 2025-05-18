# Add to utils/models.py

from adapters import AutoAdapterModel, AdapterConfig
import torch.nn as nn
from torchcrf import CRF

class AdapterWithCRF(nn.Module):
    def __init__(self, model_name, num_labels, adapter_name="ner_adapter"):
        super().__init__()
        self.adapter_name = adapter_name
        self.num_labels = num_labels
        
        # Load base model with adapter support
        self.base_model = AutoAdapterModel.from_pretrained(model_name)
        
        # Add task adapter for NER
        adapter_config = AdapterConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=16,
            non_linearity="relu",
            adapter_size=32
        )
        self.base_model.add_adapter(adapter_name, config=adapter_config)
        
        # Add classification head (without softmax/loss)
        self.base_model.add_classification_head(
            adapter_name,
            num_labels=num_labels,
            use_pooler=False
        )
        
        # Activate adapter
        self.base_model.train_adapter(adapter_name)
        
        # Add CRF layer on top
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get adapter outputs
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            adapter_names=[self.adapter_name]
        )
        
        # Get emission scores from the classification head
        emissions = outputs.logits
        
        if labels is not None:
            # Create mask for CRF (convert attention_mask to bool)
            mask = attention_mask.bool()
            
            # CRF forward pass with loss calculation
            loss = -self.crf(emissions, labels, mask=mask)
            return {"loss": loss, "emissions": emissions}
        else:
            # Decode the best path
            mask = attention_mask.bool()
            best_path = self.crf.decode(emissions, mask=mask)
            return {"predictions": best_path, "emissions": emissions}