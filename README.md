# DLC_analysis :brain:
DeepLabcut derived data analysis tools :mouse2:

![Image of mouse legend](demodata/example_results/locomouse_legend.png)

## Introduction :mouse:
The analysis tools are derived from the output of the deeplabcut based tracking .h5 file. 
The application is targeted towards locomouse data (mouse walking on beam) and mouse walking on wheel (fUS and NINscope projects). 

The code is currently maintained by [Saffira Tjon](https://neuro.nl/person/Saffira-Tjon) :pig:. 

## Contents
* **import code and plot raw data**
  * [ ] merging .h5 file per :mouse2:

* **Preprocessing**
  * [x] outlier removal
  * [x] relative to bodyaxis
  * [x] smoothing/filtering/averaging
  * [ ] movement extraction
  * [ ] weight correction
  * [ ] pixeland frames conversion

* **parameter extraction**
  * [x] stride length/duration
  * [ ] step length/duration
  * [x] bodyswing
  * [ ] velocity

Most of the code has a section that will only display upon running the file:

```

if __name__ == "__main__":

```

Here I have example usecases for the function(s) within that file. 

### Import data 
Useful functions for importing DLC data and looking at the raw traces of the label tracking. 

#### Dependecies
### Preprocessing
#### Dependecies
