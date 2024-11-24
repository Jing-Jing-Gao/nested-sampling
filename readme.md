# Coursework of Statistics for Data Science


## Description

This project involves the implementation and training of nested sampling for the lighthouse problem. It comprises two primary components:

- **Flash Location Estimation**: Using a dataset where N = 20 flash locations are stored in the first column of the file `lighthouse_flash_data.txt`, the goal is to draw stochastic samples from the posterior distribution P(α, β|{xk}) using nested sampling.
  
- **Flash Location and Intensity Estimation**: The dataset, `lighthouse_flash_data.txt`, provides flash locations and the intensity of each flash. The task involves drawing stochastic samples from the posterior distribution P (α, β, I0|{xk}, {Ik}) using nested sampling. This includes both the flash location and intensity measurements and explores the 3-dimensional posterior.

## Dataset
The `lighthouse_flash_data.txt` dataset contains two columns:
- Flash locations (first column)
- Intensity of each flash (second column)


## Utility

### Create Conda Environment:
```bash
$ conda env create -f environment.yml -n your_environment_name
```
### Activate the Conda Environment: 
```bash
$ conda activate your_environment_name
```
### Coursework Section

To measure the lighthouse location and intensity, follow these steps:

#### Lighthouse Location

To infer the lighthouse location, run the following command:

```bash
$ python src/main.py 
```

This command will execute nested sampling to obtain the posterior distribution of the lighthouse location (α, β), where α represents the position along a straight coastline and β represents the distance out to sea. The output will include:

- A 2-dimensional histogram displaying the joint posterior on α and β.
- 1-dimensional histograms of the marginalised posteriors on both parameters.
- Measurements of both parameters presented in the form mean ± standard deviation.
- Evidence of suitable convergence diagnostic information for the nested sampling algorithm.


#### Lighthouse Location and Intensity

To infer the lighthouse location, run the following command:

```bash
$ python src/main_intensity.py 
```

This command will execute nested sampling to obtain the posterior distribution of the lighthouse location and intensity (α, β, logI), where α represents the position along a straight coastline, β represents the distance out to sea, and logI characterizes the logarithmic intensity value. The output will include:

- Three 2-dimensional histograms displaying the joint posterior on α, β, and logI.
- Three 1-dimensional histograms of the marginalised posteriors on all parameters.
- Measurements of all parameters presented in the form mean ± standard deviation.
- Evidence of suitable convergence diagnostic information for the nested sampling algorithm.


## License
This project is licensed under the [MIT License](LICENSE).