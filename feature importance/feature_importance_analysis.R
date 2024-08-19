# Load necessary libraries for analysis
library(fixest)       # For fixed effects models (FEOLs)
library(etable)       # For creating tables from model outputs
library(nlme)         # For mixed-effects models
library(modelsummary) # For summarizing models
library(dplyr)        # For data manipulation
library(lmtest)       # For diagnostic tests for linear models
library(sandwich)     # For robust standard errors

# Load the dataset with visual variables
data <- read.csv("../dataset/infant_ads_variables06292024.csv")

# Define the independent variables (visual features)
ivs <- c('color_black_mean', 'color_black_std', 'color_blue_mean',
         'color_blue_std', 'color_color_diversity_std', 'color_gray_mean',
         'color_green_std', 'color_hue_mean', 'color_orange_mean',
         'color_orange_std', 'color_purple_mean', 'color_purple_std',
         'color_red_mean', 'color_red_std', 'color_saturation_mean',
         'color_saturation_std', 'color_yellow_mean', 'color_yellow_std',
         'inner_brightness_std', 'inner_sharpness_mean',
         'inner_sharpness_std', 'objects_count_yolo_std',
         'region_count_std', 'region_size_avg_mean', 'region_size_avg_std',
         'rule_of_thirds_mean', 'texture_c1_contrast_mean',
         'texture_c1_correlation_mean', 'texture_c1_correlation_std',
         'texture_c1_dissimilarity_mean', 'texture_c1_dissimilarity_std',
         'texture_c1_energy_std', 'texture_c2_contrast_std',
         'texture_c2_correlation_mean', 'texture_c2_correlation_std',
         'texture_c2_energy_mean', 'texture_c2_homogeneity_mean',
         'texture_c3_correlation_std', 'texture_c3_homogeneity_mean')

# Create the regression formula for visual features, with fixed effects for 'id'
formula <- paste("convex_hull_area ~", paste(ivs, collapse = " + "), " | id ")

# Run the fixed effects linear regression model
model <- feols(as.formula(formula), data = data)

# Display the summary of the regression model
summary(model)


# Load the dataset with linguistic variables
data <- read.csv("../dataset/infant_ads_variables_text06292024.csv")

# Define the independent variables (linguistic features)
ivs <- c('absolutist', 'acquire', 'adj', 'agency', 'agentrelatedemotions', 'allnone', 'allpunc', 'allure', 'alterations', 'authentic', 'auxverb', 'behavioral_activation', 'cognition', 'cogs_precogs', 'conj', 'dic', 'differ', 'effort_enjoyment', 'excitement', 'exclam', 'fabulations', 'feeling', 'focusfuture', 'focuspast', 'home', 'illness', 'infinity_eternity', 'insight', 'lack', 'lifestyle', 'linguistic', 'memory', 'mindoverall', 'nonflu', 'number', 'otherp', 'period', 'politic', 'relig', 'reward', 'risk', 'ruggedness', 'security', 'sexual', 'sincerity', 'sophistication', 'swear', 'tentat', 'they', 'time', 'tone', 'verb', 'wc', 'you')

# Create the regression formula for linguistic features, with fixed effects for 'id'
formula <- paste("convex_hull_area ~", paste(ivs, collapse = " + "), " | id ")

# Run the fixed effects linear regression model
model <- feols(as.formula(formula), data = data)

# Display the summary of the regression model
summary(model)


# Load the dataset with audio variables
data <- read.csv("../dataset/infant_ads_variables_audio06292024.csv")

# Define the independent variables (audio features)
ivs <- c('chroma_10_mean', 'chroma_11_mean', 'chroma_4_mean',
         'chroma_5_mean', 'chroma_7_std', 'chroma_9_mean', 'chroma_9_std',
         'mel_spectrogram_mean', 'mel_spectrogram_std', 'mfcc_10_mean',
         'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_12_std', 'mfcc_13_mean',
         'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
         'mfcc_4_mean', 'mfcc_6_std', 'mfcc_7_mean', 'mfcc_7_std',
         'mfcc_8_std', 'mfcc_9_mean', 'pitch_mean', 'pitch_std', 'rms_max',
         'rms_mean', 'rms_std', 'spectral_bandwidth_mean',
         'spectral_bandwidth_std', 'spectral_centroid_mean', 'zcr_mean')

# Create the regression formula for audio features, with fixed effects for 'id'
formula <- paste("convex_hull_area ~", paste(ivs, collapse = " + "), " | id ")

# Run the fixed effects linear regression model
model <- feols(as.formula(formula), data = data)

# Display the summary of the regression model
summary(model)
