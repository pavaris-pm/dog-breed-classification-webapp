# Dog Breed Classification
- Plan : utilizing DETR + ConvNext architecture with extra optional CVAEncoder + classifier engine,
- Detector : DETR
- Classifier : ConvNext / ConvolutionalAutoEncoder (full-architecture)
- an implementation details will be updated soon

# Web Application Demo
For Demo Web Application, You can enter it via 
[dog-breed-classification website](https://oppai-dog-breed-classification.streamlit.app/)
![landing-page](src/production/web_image/landing_page.png)


To run script in order to Open Web Application Dashboard
```
git clone https://github.com/pavaris-pm/dog-breed-classification.git
pip install -r requirements.txt
streamlit run src/production/dashboard.py
```

## Citations

If you use `ConvNextV2-Large-DogBreed` in your project or publication, please cite the model as follows

```
Pavaris Ruangchutiphophan. ConvNextV2-Large-DogBreed (Revision 3ddd523). Huggingface, 2023, http://doi.org/10.57967/hf/1288 
```

or BibTeX entry:

```bib
@misc {pavaris_ruangchutiphophan_2023,
	author       = { {Pavaris Ruangchutiphophan} },
	title        = { ConvNextV2-large-DogBreed (Revision 3ddd523) },
	year         = 2023,
	url          = { https://huggingface.co/Pavarissy/ConvNextV2-large-DogBreed },
	doi          = { 10.57967/hf/1288 },
	publisher    = { Hugging Face }
}
```
