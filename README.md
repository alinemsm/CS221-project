# Customised LLMs from Specialised Documents

This project aims to create customised, domain-specific language models by fine-tuning small, open-source models using curated domain knowledge from specialised documents. The focus is on the field of data science, leveraging high-quality textbooks to generate synthetic question-answer datasets for fine-tuning.

## Project Overview

- Base model: Gemma-2b-it
- Fine-tuning approach: Low Rank Adaptation (LoRA)
- Dataset generation: Claude-3 Opus to create synthetic Q&As
- Evaluation metrics: ROUGE-L scores

## Repository Structure

- `data/`: Contains the generated question-answer dataset and prompt
- `src/`: Contains the source code for data generation and evaluation
- `notebooks/`: Jupyter notebooks for fine-tuning and evaluation in Colab
- `output/`: Contains all the evaluation results

## Getting Started

1. Clone the repository:

    ```bash
   git clone https://github.com/alinemsm/CS221-project.git
    ```
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Generate the synthetic question-answer dataset:

    ```bash
    python src/data_generation.py
    ```

4. Fine-tune the base model:

   ```jupyter
   notebooks/colab_tuning_ft.ipynb
   notebooks/colab_tuning_f2xh.ipynb
   ```

5. Evaluate the fine-tuned model:

   ```jupyter
   notebooks/evaluation_chatgpt.ipynb
   ```

6. Fine-tuned model checkpoints [here](https://drive.google.com/drive/folders/1Dt2HY7cDjHxQjYQixyQAblB-CcZs2Bki?usp=share_link).


## Acknowledgements
This project benefited from the guidance and support of the Stanford CS221 course staff, including the teaching team and the course TAs, specially Rohan Taori.

## References

- [Gemma-2b-it](https://www.kaggle.com/models/keras/gemma/Keras/gemma_2b_en/1)
- [Fine-tuning Gemma models in Keras using LoRA](https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora)
- [GoogleCloudPlatform - vertex-ai-samples](notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

## Contributing

We welcome contributions to this project. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [Aline Menezes](mailto:alinemsm@stanford.edu).