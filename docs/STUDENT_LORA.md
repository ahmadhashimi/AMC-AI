# Student LoRA plan

LoRA is designed to adapt a **trained** base model economically by freezing its original weights and training low-rank additions to selected linear layers. AMC's tiny educational checkpoint is not useful as a student model. Do not distribute it as if it can tutor students.

After AMC trains and validates a capable checkpoint, publish:
1. Compatible Transformers/vLLM model exports and model card
2. A PEFT LoRA notebook using `LoraConfig`, `get_peft_model`, and supervised examples
3. A dataset card for each student project
4. A GPU-memory and compute guide with cloud budget controls
5. Evaluation before and after fine-tuning

Students can use LoRA/QLoRA without modifying the original base weights. Data license and model license compatibility remain essential.
