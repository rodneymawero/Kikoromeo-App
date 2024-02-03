# 📦 Kikoromeo-App

Hypothyroid disease prediction, using the Thyroid Dataset from the UCI Machine Learning Repository. Courtesy of the Garavan Institute and Ross Quinlan

## Kikoromeo-App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kikoromeo-app-uhjtjqghrwavasvsbmcasi.streamlit.app/)

## 1.0 Introduction

The thyroid hormones are involved in important functions in the body. For example , the hormones control how the cells of the body use energy to sustain life (metabolism). The thyroid hormones also regulate body temperature, digestion, breathing and heart rate. These life processes speed up as thyroid hormone levels rise. But problems occur if the thyroid makes too much hormone(hyperthyrodism) or not enough (hypothyrodism) leading to thyroid disorders.

Goitre, for instance, can be caused by deficiency of iodine in the diet.  The body needs iodine to produce thyroid hormone. If you do not have enough iodine in your diet, the thyroid gets larger to try and capture all the iodine it can, so it can make the right amount of thyroid hormone. So, goitre can be a sign the thyroid is not able to make enough thyroid hormone. The use of iodized salt prevents iodine deficiency.

Thyroid problems are also often caused by autoimmune disorders, in which the immune system mistakenly attacks and destroys the body’s own cells. For example, an autoimmune disorder called Graves’ disease can cause the thyroid to be over-active, while one called Hashimoto’s disease can make the thyroid under-active.

Hypothyroidism is a condition where the thyroid gland is underactive and doesn't produce enough thyroid hormones. The primary hormones produced by the thyroid gland are Triiodothyronine (T3), Thyroxine (T4), and Thyroid-Stimulating Hormone (TSH). Testing the levels of these hormones in the body is important in determining whether one is hypothyroid or not.

### 1.1 Problem Statement

Hypothyrodism can be hard to diagnose. First, its symptoms are similar to other diseases. These include fatigue, losing weight,

fast heartbeat(tachycardia), irregular heartbeat (arrhythmia), enlarged thyroid gland, sometimes called a goiter, increased hunger, nervousness, changes in menstrual cycle, anxiety and irritability among others.
https://www.mayoclinic.org/diseases-conditions/hypothyroidism/symptoms-causes/syc-20350284

Many patients mistakenly attribute syptoms to aging or other factors and unnecessarily delay treatment https://newsinhealth.nih.gov/2015/09/thinking-about-your-thyroid#:~:text=When%20thyroid%20glands%20don't,in%20people%20over%20age%2060.

Secondly, the disease can develop over months or even years, hence symptoms are often hard to pick up on. The objectives of this project is to build a machine learning model that will:

-1. Predict whether a patient is hypothyroid or not,

-2. Detemine the type of hypothyrodism based on clinical data.

-3. Deploy an app that will not only display the class, but the prediction probabilities for the given class.

## The Model Deployed

The model used in deployment is the Catboost model trained with the OneVsRestClassifier strategy for multi-class classification tasks in Scikit-learn. The Catboost_clf not only predicts whether one is negative, primary hypothyroid or has compensatory hypothyrodism, but also outputs the prediction probabilities for the predicted class, given the user inputs of course ;-).

CatBoost is  known for its robustness and ability to handle categorical features. The OneVsRestClassifier strategy in scikitlearn  extends binary classification to multi-class scenarios. This combination allows one to effectively tackle complex classification problems with superior performance and accuracy.
