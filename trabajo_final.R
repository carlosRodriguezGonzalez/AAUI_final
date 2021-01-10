install.packages("e1071") 
library(C50)
library(caret)
library(quanteda)
library(tidyverse)
library(tidytext)

library(e1071)

library(tm)



system("sed 's/;||;/\t/g' ./dataset/labeled_corpus_6K.txt > newfakedata.txt")


setwd("/Users/sergioredondo/Desktop/UTAD/U-TAD_Alumno/4_Cuarto/aprendizaje_automatico_1/00_trabajo_final/")
#data <-read.delim("labeled_corpus_6K.txt", header = FALSE, sep = "|")
#datos <-read.table("labeled_corpus_6K.txt", header = FALSE, sep = "\t", dec = ";")
datos <- read.csv("./dataset/HateSpeechData.csv", encoding="UTF-8", header = FALSE, sep = "\t")

summary(datos)
apply(is.na(datos),2,sum)
which(is.na(datos$V3))
datos$V2[4]
#na.omit(datos)

head(datos)

###
df <- setNames(datos,   c("id", "text", "class"))
df$class <- as.factor(df$class)
summary(df)
str(df)
head(df)

###
df$text[1]

df$text <- gsub("[\U0001F600-\U0001F64F]+", "", df$text)

df$text[1]

###
df$text[1]

df$text <- tolower(df$text)

df$text[1]

###

df$text[1858]
df$text <- gsub("[!^á]", "a", df$text)
df$text[1858]

df$text[1874]
df$text <- gsub("[!^é]", "e", df$text)
df$text[1874]

df$text[1874]
df$text <- gsub("[!^í]", "i", df$text)
df$text[1874]

df$text[1874]
df$text <- gsub("[!^ó]", "o", df$text)
df$text[1874]

df$text[1868]
df$text <- gsub("[!^ú]", "u", df$text)
df$text[1868]


###

df$text[17]

df$text <- gsub('(f|ht)tp\\S+\\s*',"", df$text)

df$text[17]

###

df$text[9]
# removal of @name[mention]
df$text <- gsub('@[a-zA-Z0-9]{1,15}',"", df$text)

df$text[9]

###

df$text[39]

df$text <- gsub('#',"", df$text)

df$text[39]

###

df$text[48]
# replace normal numbers with numbr
df$text <- gsub('\\d+(\\.\\d+)?',"numbr", df$text)

df$text[48]

###

df$text[9]
# remove leading and trailing whitespace
df$text <- gsub('^\\s+|\\s+?$',"", df$text)

df$text[9]

###

df$text[39]
# remove whitespace with a single space
df$text <- gsub('\\s+'," ", df$text)

df$text[39]

###
# Creamos un corpus con el contenido de los tuits
hate <- corpus(df$text[df$class==1])
not_hate <- corpus(df$text[df$class==0])

length(hate)
summary(hate)
class(hate)

length(not_hate)
summary(not_hate)
class(not_hate)


sparse_hate <- dfm(hate, 
                 remove = stopwords("spanish"), 
                 stem = FALSE, 
                 remove_punct = TRUE)

topfeatures(sparse_hate, 100)  # 20 top words


sparse_not_hate <- dfm(not_hate, 
                   remove = stopwords("spanish"), 
                   stem = FALSE, 
                   remove_punct = TRUE)

topfeatures(sparse_not_hate, 100)  # 20 top words

###
additional_stopwords <- c("rt",
                          "lol",
                          "q",
                          "d"
)

mystopwords <- c(stopwords("spanish"), additional_stopwords)

sparse_hate <- dfm(hate, 
                   remove = mystopwords, 
                   stem = FALSE, 
                   remove_punct = TRUE)

topfeatures(sparse_hate, 100)  # 20 top words


sparse_not_hate <- dfm(not_hate, 
                       remove = mystopwords, 
                       stem = FALSE, 
                       remove_punct = TRUE)

topfeatures(sparse_not_hate, 100)  # 20 top words

###

set.seed(100)
textplot_wordcloud(sparse_hate, 
                   min_count = 25, 
                   random_order = FALSE,
                   rotation = .25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))



textplot_wordcloud(sparse_not_hate, 
                   min_count = 80, 
                   random_order = FALSE,
                   rotation = .25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

###

mytoks <- 
  tokens(hate, 
         remove_punct = TRUE,
         remove_numbers = TRUE) %>%
  tokens_remove(mystopwords, padding = TRUE)

mytoks

mytoks_2 <- tokens_ngrams(mytoks, n = 2, 
                          concatenator = "-")
print(mytoks_2)

quant_dfm <- dfm(mytoks_2)

quant_dfm <- dfm_trim(quant_dfm, min_termfreq = 2)
quant_dfm

topfeatures(quant_dfm, 100)  # 100 top bigrams

textplot_wordcloud(quant_dfm, 
                   min_count = 2, 
                   random_order = FALSE,
                   rotation = 0, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))
###

###

mytoks <- 
  tokens(not_hate, 
         remove_punct = TRUE,
         remove_numbers = TRUE) %>%
  tokens_remove(mystopwords, padding = TRUE)

mytoks

mytoks_2 <- tokens_ngrams(mytoks, n = 2, 
                          concatenator = "-")
print(mytoks_2)

quant_dfm <- dfm(mytoks_2)

quant_dfm <- dfm_trim(quant_dfm, min_termfreq = 2)
quant_dfm

topfeatures(quant_dfm, 100)  # 100 top bigrams

textplot_wordcloud(quant_dfm, 
                   min_count = 2, 
                   random_order = FALSE,
                   rotation = 0, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))
###















#train_index <- sample(1:nrow(datos), 0.7 * nrow(datos))
#test_index <- setdiff(1:nrow(datos), train_index)

#X_train <- as.data.frame(datos[1:100, -2])
#y_train <- datos[1:100, "V3"]

#X_test <- as.data.frame(datos[101:131, -2])
#y_test <- datos[101:131, "V3"]


#modelo_logit<- glm(y_train ~. , data = X_train, family = binomial(link = "logit"))

#summary(modelo_logit)

#pred_logit <- predict(modelo_logit, newdata= datos[101:30, -2], type="response")

#confusionMatrix(table(as.character(pred_logit), as.character(y_test)), positive="1")

#plotROC(y_test, pred_logit)

#modelo3 <- C5.0(default~., data=X_train)
#summary(modelo3)
#plot(modelo3)

#prediccion <- predict(modelo3, X_test)
#confusionMatrix(prediccion, y_test)






###################

data_counts <- map_df(1:2,
                      ~ unnest_tokens(df, word, text, 
                                      token = "ngrams", n = .x)) %>%
  anti_join(stop_words, by = "word") %>%
  count(id, word, sort = TRUE)

words_10 <- data_counts %>%
  group_by(word) %>%
  summarise(n = n()) %>% 
  filter(n >= 10) %>%
  select(word)

data_dtm <- data_counts %>%
  right_join(words_10, by = "word") %>%
  bind_tf_idf(word, id, n) %>%
  cast_dtm(id, word, tf_idf)

meta <- tibble(id = dimnames(data_dtm)[[1]]) %>%
  left_join(df[!duplicated(df$id), ], by = "id")


set.seed(1234)
trainIndex <- createDataPartition(meta$class, p = 0.8, list = FALSE, times = 1)

data_df_train <- data_dtm[trainIndex, ] %>% as.matrix() %>% as.data.frame()
data_df_test <- data_dtm[-trainIndex, ] %>% as.matrix() %>% as.data.frame()

response_train <- meta$class[trainIndex]
response_test <- meta$class[-trainIndex]


trctrl <- trainControl(method = "none")

# SVM
svm_mod <- train(x = data_df_train,
                 y = as.factor(response_train),
                 method = "svmLinearWeights2",
                 trControl = trctrl,
                 tuneGrid = data.frame(cost = 1, 
                                       Loss = 0, 
                                       weight = 1))

svm_pred <- predict(svm_mod,
                    newdata = data_df_test)

svm_cm <- confusionMatrix(svm_pred, response_test)
svm_cm


# Naive bayes
nb_mod <- train(x = data_df_train,
                y = as.factor(response_train),
                method = "naive_bayes",
                trControl = trctrl,
                tuneGrid = data.frame(laplace = 0,
                                      usekernel = FALSE,
                                      adjust = FALSE))

nb_pred <- predict(nb_mod,
                   newdata = data_df_test)

nb_cm <- confusionMatrix(nb_pred, response_test)
nb_cm


# LOGIT
logitboost_mod <- train(x = data_df_train,
                        y = as.factor(response_train),
                        method = "LogitBoost",
                        trControl = trctrl)

logitboost_pred <- predict(logitboost_mod,
                           newdata = data_df_test)
logitboost_cm <- confusionMatrix(logitboost_pred, response_test)
logitboost_cm


# Random forest

rf_mod <- train(x = data_df_train, 
                y = as.factor(response_train), 
                method = "ranger",
                trControl = trctrl,
                tuneGrid = data.frame(mtry = floor(sqrt(dim(data_df_train)[2])),
                                      splitrule = "gini",
                                      min.node.size = 1))
rf_pred <- predict(rf_mod,
                   newdata = data_df_test)
rf_cm <- confusionMatrix(rf_pred, response_test)
rf_cm


# NNet

nnet_mod <- train(x = data_df_train,
                  y = as.factor(response_train),
                  method = "nnet",
                  trControl = trctrl,
                  tuneGrid = data.frame(size = 1,
                                        decay = 5e-4),
                  MaxNWts = 5000)
nnet_pred <- predict(nnet_mod,
                     newdata = data_df_test)
nnet_cm <- confusionMatrix(nnet_pred, response_test)
nnet_cm


### 
mod_results <- rbind(
  svm_cm$overall, 
  nb_cm$overall,
  logitboost_cm$overall,
  rf_cm$overall,
  nnet_cm$overall
) %>%
  as.data.frame() %>%
  mutate(model = c("SVM", "Naive-Bayes", "LogitBoost", "Random forest", "Neural network"))

mod_results %>%
  ggplot(aes(model, Accuracy)) +
  geom_point() +
  ylim(0, 1) +
  geom_hline(yintercept = mod_results$AccuracyNull[1],
             color = "red")


#


fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 3,
                           search = "grid")
svm_mod <- train(x = data_df_train,
                 y = as.factor(response_train),
                 method = "svmLinearWeights2",
                 trControl = fitControl,
                 tuneGrid = data.frame(cost = 0.01, 
                                       Loss = 0, 
                                       weight = seq(0.5, 1.5, 0.1)))
plot(svm_mod)
svm_pred <- predict(svm_mod,
                    newdata = data_df_test)

svm_cm <- confusionMatrix(svm_pred, response_test)
svm_cm
