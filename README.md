# Q2_Model_Deployment

# 1. Reason for model selection - >

I chose DistilBERT base uncased distilled SQuAD as my Huggingface model. DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model. More precisely, it was pretrained with three objectives:

i. Distillation loss: the model was trained to return the same probabilities as the BERT base model.

ii. Masked language modeling (MLM): this is part of the original training loss of the BERT base model. When taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

iii. Cosine embedding loss: the model was also trained to generate hidden states as close as possible as the BERT base model.
This way, the model learns the same inner representation of the English language than its teacher model, while being faster for inference or downstream tasks.


# 2. Model_Deployment_Docker.ipynb: Post to Container endpoint & prints out the response - >
    
  i. Install Sagemaker SDK, Transformers in Python3
  ii. Install necessary Libraries
  iii. Create SageMaker endpoint with the chosen model
  iv.  Deploy model to SageMaker Inference
  v. Test with Sample Input
  vi. Define Predictor
  
 # 3. Dockerfile - >

   i. Pull image from Dockerhub
   ii. Set Server components to support multiple parallel incoming requests
   iii. Install necessary dependencies for SageMaker Inference Toolkit
   iv.  Install pip requirements
    v.  Install Hugging Face libraries and its dependencies
    vi.  Copy entrypoint script to the image
    vii. Copy the default custom service file to handle incoming data and inference requests
    viii. Define an entrypoint script for the docker image
    ix. Define command to be passed to the entrypoint
     
