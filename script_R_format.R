###########################################################################################
###########################   MovieLens Project  ##########################################
###########################################################################################


############### Create edx set and validation set #########################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset web location:
### https://grouplens.org/datasets/movielens/10m/
### http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

### For R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Partitioning the data intro edx and validation set. Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Making sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

########################## Building the recommendation system ############################

##### Partitioning the edx data into train and test. Test is the 10% of the edx dataset

set.seed(1, sample.kind="Rounding")

index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-index,]
test <- edx[index,]

rm(index)# remove index

test <- test %>% 
        semi_join(train, by = "movieId") %>%
        semi_join(train, by = "userId") # make sure we don’t include users and movies in the test set that do not appear in the training set

##### Inspecting the train data (ten users and nine movies)

n_user_movie <- train %>%   summarize(number_users = n_distinct(userId),
            number_movies = n_distinct(movieId)) %>%
            knitr::kable()# number of users that provide ratings and unique movies in the train dataset

movie_top5 <- train %>%
              dplyr::count(movieId) %>%
              top_n(5) %>%
              pull(movieId) # identifying the top 5 movies

user_movie_tab <- train %>%
                  filter(userId %in% c(1:20)) %>% 
                  filter(movieId %in% movie_top5) %>% 
                  select(userId, title, rating) %>% 
                  spread(title, rating) # identifying user's rating for four movies

user_movie_tab %>% knitr::kable() # summary


train %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies") # distribution of movies receiving ratings 

train %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users") # distribution of user providing ratings 


##### Building prediction models 

### model 1: just the average ratings for all movies and users 

mu <- mean(train$rating) 
mu

m1_rmse <- RMSE(test$rating, mu)
m1_rmse

rmse_models <- tibble(method = "Model 1: Average ratings", RMSE = m1_rmse) # summary model 1
rmse_models %>% knitr::kable()

### model 2: accounting for movie effects

movie_avg <- train %>% 
            group_by(movieId) %>% 
            summarize(b_movie = mean(rating - mu))

m2_p_ratings <- mu + test %>% 
                    left_join(movie_avg, by='movieId') %>%
                    pull(b_movie)

m2_rmse <- RMSE(m2_p_ratings, test$rating)
m2_rmse

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 2: Movie effect",
                                    RMSE = m2_rmse)) # summary 
rmse_models %>% knitr::kable()


### model 3: accounting for movie and user effects

user_avg <- train %>% 
  left_join(movie_avg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))

m3_p_ratings <- test %>% 
                left_join(movie_avg, by='movieId') %>%
                left_join(user_avg, by='userId') %>%
                mutate(pred = mu + b_movie + b_user) %>%
                pull(pred)

m3_rmse <- RMSE(m3_p_ratings, test$rating)
m3_rmse

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 3: Movie + user effect",
                                RMSE = m3_rmse)) # summary 
rmse_models %>% knitr::kable()


### model 4: accounting for movie, user and day effects

library(lubridate)
train <- mutate(train, date = as_datetime(timestamp))
train <-  mutate(train, day = round_date(date, unit = "day"))
test <- mutate(test, date = as_datetime(timestamp))
test <-  mutate(test, day = round_date(date, unit = "day"))

day_avg <- train %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  group_by(day) %>%
  summarize(b_day = mean(rating - mu - b_movie - b_user))

m4_p_ratings <- test %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  mutate(pred = mu + b_movie + b_user + b_day) %>%
  pull(pred)

m4_rmse <- RMSE(m4_p_ratings, test$rating)
m4_rmse

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 4: Movie + user + day effect",
                                RMSE = m4_rmse)) # summary 
rmse_models %>% knitr::kable()


### model 5: accounting for movie, user, day and week effects

train <-  mutate(train, week = round_date(date, unit = "week"))
test <-  mutate(test, week = round_date(date, unit = "week"))

week_avg <- train %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  group_by(week) %>%
  summarize(b_week = mean(rating - mu - b_movie - b_user - b_day))

m5_p_ratings <- test %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  left_join(week_avg, by='week') %>%
  mutate(pred = mu + b_movie + b_user + b_day + b_week) %>%
  pull(pred)

m5_rmse <- RMSE(m5_p_ratings, test$rating)
m5_rmse

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 5: Movie + user + day + week effect",
                                RMSE = m5_rmse)) # summary 

rmse_models %>% knitr::kable() # there is no improvement on the model so week is no considered anymore


### model 6: accounting for movie, user, day and month effects

train <-  mutate(train, month = round_date(date, unit = "month"))
test <-  mutate(test, month = round_date(date, unit = "month"))

month_avg <- train %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  group_by(month) %>%
  summarize(b_month = mean(rating - mu - b_movie - b_user - b_day))

m6_p_ratings <- test %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  left_join(month_avg, by='month') %>%
  mutate(pred = mu + b_movie + b_user + b_day + b_month) %>%
  pull(pred)

m6_rmse <- RMSE(m6_p_ratings, test$rating)
m6_rmse

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 6: Movie + user + day + month effect",
                                RMSE = m6_rmse)) # summary 

rmse_models %>% knitr::kable() # no month effect so it is not used any further


### model 7: accounting for movie, user, day and year effects

train <-  mutate(train, year = round_date(date, unit = "year"))
test <-  mutate(test, year = round_date(date, unit = "year"))

year_avg <- train %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  group_by(year) %>%
  summarize(b_year = mean(rating - mu - b_movie - b_user - b_day))

m7_p_ratings <- test %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(day_avg, by='day') %>%
  left_join(year_avg, by='year') %>%
  mutate(pred = mu + b_movie + b_user + b_day + b_year) %>%
  pull(pred)

m7_rmse <- RMSE(m7_p_ratings, test$rating)
m7_rmse 

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 7: Movie + user + day + year effect",
                                RMSE = m7_rmse)) # summary 

rmse_models %>% knitr::kable() # year does not show any improvement so it is not considered any further


### Regularisation 
### Regularization permits us to penalize large estimates that are formed using small sample sizes

# Using only the movie effects below it is shown the top 10 worst and best movies based on
# the movie efect (b_movie) and number of ratings. As showed the supposed “best” and “worst” movies 
# were rated by very few users, in most cases just 1.

movie_titles <- train %>% 
  select(movieId, title) %>%
  distinct()

train %>% dplyr::count(movieId) %>% 
  left_join(movie_avg) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_movie)) %>% 
  select(title, b_movie, n) %>% 
  slice(1:10) %>% 
  knitr::kable() # top 10 best movies 

train %>% dplyr::count(movieId) %>% 
  left_join(movie_avg) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_movie) %>% 
  select(title, b_movie, n) %>% 
  slice(1:10) %>% 
  knitr::kable() # top 10 worst movies 


## Penalized least squares

# Model 2 regularisation: movie effect. 
# Using cross-validation to choose the penalty term (lambda)

lambdas <- seq(0, 10, 0.25)

m2_rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(n()+l))
  m2_reg_p_ratings <- test %>% 
    left_join(b_movie, by = "movieId") %>%
    mutate(pred = mu + b_movie) %>%
    pull(pred)
  return(RMSE(m2_reg_p_ratings, test$rating))
})

qplot(lambdas, m2_rmses)  

m2_lambda <- lambdas[which.min(m2_rmses)]
m2_lambda

# Computing the regularised estimates of  b_movie using m2_lambda

movie_reg_avg <- train %>% 
  group_by(movieId) %>% 
  summarize(b_movie = sum(rating - mu)/(n()+m2_lambda), n_i = n()) 

# plotting the regularised estimates versus the least squares estimates to see how they shrink

tibble(Least_squares_estimates = movie_avg$b_movie, 
       Regularised_estimates = movie_reg_avg$b_movie, 
       n = movie_reg_avg$n_i) %>%
  ggplot(aes(Least_squares_estimates, Regularised_estimates, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) # when n is small the values shrink towards zero

# the top 10 best and worst movies based on the penalised estimates

train %>%
  count(movieId) %>% 
  left_join(movie_reg_avg, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_movie)) %>%
  slice(1:10) %>% 
  pull(title) # top 10 best movies

train %>%
  count(movieId) %>% 
  left_join(movie_reg_avg, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_movie) %>% 
  select(title, b_movie, n) %>% 
  slice(1:10) %>% 
  pull(title)

# Checking the new RMSE

min(m2_rmses)

# binding all the models together for comparison

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 2 reg: Movie effect regularisation",
                                RMSE = min(m2_rmses)))
rmse_models %>% knitr::kable()


# Model 3 regularization: movie and user effect. 
# Using cross-validation to choose the penalty term (lambda)

m3_rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_movie <- train %>%
            group_by(movieId) %>%
            summarize(b_movie = sum(rating - mu)/(n()+l))
  b_user <- train %>% 
            left_join(b_movie, by="movieId") %>%
            group_by(userId) %>%
            summarize(b_user = sum(rating - b_movie - mu)/(n()+l))
  m3_reg_p_ratings <- test %>% 
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(pred = mu + b_movie + b_user) %>%
    pull(pred)
  return(RMSE(m3_reg_p_ratings, test$rating))
})

qplot(lambdas, m3_rmses)  

m3_lambda <- lambdas[which.min(m3_rmses)]
m3_lambda

# Computing the regularised estimates of b_user using m3_lambda
user_reg_avg <- train %>% 
  left_join(movie_reg_avg, by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_user = sum(rating - b_movie - mu)/(n()+m3_lambda), n_i = n())

# plotting the regularised estimates versus the least squares estimates to see how they shrink

tibble(Least_squares_estimates = user_avg$b_user, 
       Regularised_estimates = user_reg_avg$b_user, 
       n = user_reg_avg$n_i) %>%
  ggplot(aes(Least_squares_estimates, Regularised_estimates, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) # when n is small the values shrink towards zero

# Checking the new RMSE

min(m3_rmses)

# binding all the models together for comparison

rmse_models <- bind_rows(rmse_models,
                        tibble(method="Model 3 reg: Movie + user effect regularisation",
                                    RMSE = min(m3_rmses)))
rmse_models %>% knitr::kable()



# Model 4 regularization: movie, user and day effect. 
# Using cross-validation to choose the penalty term (lambda)

m4_rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(n()+l))
  b_user <- train %>% 
    left_join(b_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - b_movie - mu)/(n()+l))
  b_day <- train %>% 
    left_join(b_movie, by="movieId") %>%
    left_join(b_user, by="userId") %>%
    group_by(day) %>%
    summarize(b_day = sum(rating - b_movie - b_user - mu)/(n()+l))
  m4_reg_p_ratings <- test %>% 
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_day, by = "day") %>%
    mutate(pred = mu + b_movie + b_user + b_day) %>%
    pull(pred)
  return(RMSE(m4_reg_p_ratings, test$rating))
})

qplot(lambdas, m4_rmses)  

m4_lambda <- lambdas[which.min(m4_rmses)]
m4_lambda

# Computing the regularised estimates of b_user using m4_lambda
day_reg_avg <- train %>% 
  left_join(movie_reg_avg, by='movieId') %>%
  left_join(user_reg_avg, by='userId') %>%
  group_by(day) %>% 
  summarize(b_day = sum(rating - b_movie - b_user - mu)/(n()+m4_lambda), n_i = n())

# plotting the regularised estimates versus the least squares estimates to see how they shrink

tibble(Least_squares_estimates = day_avg$b_day, 
       Regularised_estimates = day_reg_avg$b_day, 
       n = day_reg_avg$n_i) %>%
  ggplot(aes(Least_squares_estimates, Regularised_estimates, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) # when n is small the values shrink towards zero


# Checking the new RMSE

min(m4_rmses)

# binding all the models together for comparison

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Model 4 reg: Movie + user + day effect regularisation",
                                RMSE = min(m4_rmses)))
rmse_models %>% knitr::kable()


# Validation with the best performing model

## adding the day to the validation dataset

validation <- mutate(validation, date = as_datetime(timestamp))
validation <-  mutate(validation, day = round_date(date, unit = "day"))

# making sure that the validation set has the same information than the train set

validation <- validation %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId") %>%
  semi_join(train, by = "day")


validation_pred <- validation %>% 
  left_join(movie_reg_avg, by = "movieId") %>%
  left_join(user_reg_avg, by = "userId") %>%
  left_join(day_reg_avg, by = "day") %>%
  mutate(pred = mu + b_movie + b_user + b_day)

validation_rmse <- RMSE(validation_pred$pred, validation$rating) 

rmse_models <- bind_rows(rmse_models,
                         tibble(method="Validation",
                                RMSE = min(validation_rmse)))
rmse_models %>% knitr::kable()

