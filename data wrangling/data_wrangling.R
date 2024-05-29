require(tidycensus)
require(censusapi)
require(dplyr)
require(tidyr)
require(ggplot2)
require(writexl)

county_data_2016 <- read.csv("county_facts.csv")
county_facts_dictionary <- read.csv("county_facts_dictionary.csv")
county_results <- read.csv("US_County_Level_Presidential_Results_12-16.csv")

county_data_2016 <- as.data.frame(county_data_2016)
county_facts_dictionary <- as.data.frame(county_facts_dictionary)
county_results <- as.data.frame("US_County_Level_Presidential_Results_12-16.csv")

colnames(county_data_2016)[4:54] <- county_facts_dictionary$description

results <- data.frame(county_results$county_name,county_results$per_dem_2016, county_results$per_gop_2016, county_results$per_dem_2012, county_results$per_gop_2012)

full_data <- county_data_2016 %>%
  inner_join(results,
             by = c("area_name" = "county_results.county_name"))

write_xlsx(full_data, "data_2012_2016.xlsx")