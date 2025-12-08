# Analyzing the Factors Influencing Restaurant Ratings in Los Angeles
This project investigates how restaurant features and neighborhood demographics influence restaurant ratings in Los Angeles. Using the Yelp Fusion API, it gathers restaurant data (ratings, cuisine types, price levels, review counts, and coordinates). These data are later merged with Los Angeles demographic and geographic datasets to explore how local characteristics such as neighborhood diversity, density, and location contribute to restaurant success.

# Data sources
| Data Source | Name / Description | Source URL | Type | List of Fields | Format | Accessed with Python? | Estimated Size |
|--------------|-------------------|-------------|------|----------------|---------|------------------------|----------------|
| **1** | Yelp Fusion API – Los Angeles Restaurants | [https://docs.developer.yelp.com/docs/yelp-api](https://docs.developer.yelp.com/docs/yelp-api) | API | name, rating, price, categories, review_count, zipcode, hours, coordinates | JSON → CSV | ✅ yes | 13000+ |
| **2** | LA Cities and Zipcodes | [https://www.zipcode.com.ng/2022/06/los-angeles-zip-codes.html](https://www.zipcode.com.ng/2022/06/los-angeles-zip-codes.html) | Web | city_community, zipcode |  HTML → CSV | ✅ yes |144 cities and communities |
| **3** | Racial/Ethnic Composition – Los Angeles County | [[https://www.census.gov/data/developers/data-sets/acs-5year.html](https://www.laalmanac.com/population/po38.php)]([https://www.census.gov/data/developers/data-sets/acs-5year.html](https://www.laalmanac.com/population/po38.php)) | Web | city_community, total_population, pop_american_indian_alaska_native, pop_asian, pop_black_african_american, pop_native_hawaiian_pacific_islander, pop_white_non_hispanic, pop_some_other_race, pop_two_or_more_races, pop_hispanic_or_latino | HTML → CSV | ✅ yes | 529 |

# Results 
Restaurant ratings in Los Angeles are primarily driven by restaurant-level factors such as price level, popularity, and cuisine type, as demonstrated by feature importance plots and correlation matrices. Neighborhood demographic variables show minimal influence at the individual restaurant level, but become meaningful at the ZIP-code level, where diversity and restaurant density moderately predict higher average ratings. Geographic visualizations reveal clear spatial clustering, with several LA regions consistently outperforming others in dining quality. Overall, the analysis shows that both micro-level restaurant characteristics and macro-level neighborhood environments contribute to restaurant performance.

# Installation
- _describe what API keys, user must set where (in .enve) to be able to run the project._
  YELP_API_KEY=""
- _describe what special python packages you have used_
  listed in requirements.txt

# Running analysis 

From `src/` directory run:

`python main.py `

Results will appear in `results/` folder. All obtained will be stored in `data/`
