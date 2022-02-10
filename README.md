The methods from the following paper has been reproduced in python to the extent possible with some additional analysis of evolutionary trends. The repository contains scripts for data collection, cleaning, and analysis. 

>Mauch Matthias, MacCallum Robert M., Levy Mark and Leroi Armand M. 2015 The evolution of popular music: USA 1960–2010R. Soc. open sci.2150081150081
>https://doi.org/10.1098/rsos.150081

## Prerequisites
All the dependencies can be installed using the following command:
```
pip install -r requirements.txt
```
The scripts have to be run in the following order to replicate the analysis:

1. get_spotify_previews.py

2. get_t_lex.py

3. get_h_lex.py

4. topic_modelling.py

5. evolution_of_topics.py



