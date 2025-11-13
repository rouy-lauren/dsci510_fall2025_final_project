# Analyzing the Factors Influencing Restaurant Ratings in Los Angeles
This project investigates how restaurant features and neighborhood demographics influence restaurant ratings in Los Angeles. Using the Yelp Fusion API, it gathers restaurant data (ratings, cuisine types, price levels, review counts, and coordinates). These data are later merged with Los Angeles demographic and geographic datasets to explore how local characteristics such as neighborhood diversity, density, and location contribute to restaurant success.

# Data sources
| Data Source | Name / Description | Source URL | Type | List of Fields | Format | Accessed with Python? | Estimated Size |
|--------------|-------------------|-------------|------|----------------|---------|------------------------|----------------|
| **1** | Yelp Fusion API – Los Angeles Restaurants | [https://docs.developer.yelp.com/docs/yelp-api](https://docs.developer.yelp.com/docs/yelp-api) | API | name, rating, price, categories, review_count, zipcode, hours, coordinates | JSON → CSV | ✅ yes | ~2,000 |
| **2** | Google Maps Places API – Restaurant Data in Los Angeles | [https://developers.google.com/maps/documentation/places/web-service](https://developers.google.com/maps/documentation/places/web-service) | API | name, rating, price, categories, review_count, zipcode, hours | JSON / CSV | ❌ no | ~2,000 |
| **3** | Racial/Ethnic Composition – Los Angeles County | [https://www.census.gov/data/developers/data-sets/acs-5year.html](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Web | city_name, total_population, percent_white, percent_black, percent_asian, percent_hispanic | HTML → CSV | ✅ yes | ~100+ cities and communities |

# Results 
_describe your findings_

# Installation
- _describe what API keys, user must set where (in .enve) to be able to run the project._
  YELP_API_KEY=""
  future need API:Google
- _describe what special python packages you have used_
  listed in requirements.txt

# Running analysis 
_update these instructions_


From `src/` directory run:

`python main.py `

Results will appear in `results/` folder. All obtained will be stored in `data/`