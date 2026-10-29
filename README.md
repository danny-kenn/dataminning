# Does Class Attendance Affect Academic Performance? What Really Drives Exam Scores

**Course:** COSC391 Data Mining, University of Eastern Africa, Baraton
**Group:** John Kamau, Lynn Bitok, David Kiamba
**Live dashboard:** `https://YOURUSERNAME.github.io/student-attendance-performance/` *(replace with our GitHub Pages link once it is live)*

---

## 1. What this project is

Many lecturers tell students that attending class is the key to good marks. We wanted to test that claim with data instead of opinion. Our question was:

1. Does class attendance correlate with exam performance?
2. If it does not (or only weakly), what factors affect exam scores the most?

To answer it we used a supervised **regression** model, which predicts a number (the exam score out of 100), together with correlation analysis. Everything in this repository comes from running the code on our dataset, and every number reported in our document can be reproduced by running the scripts.

## 2. The two assignments and what we did

**Assignment 1 (Part 1)** asked each group of three to write a document of at least four paragraphs that gives:
- a well-defined title and the names of the three developers,
- a background or problem statement (at least two paragraphs) explaining the problem and why a data mining model is a good solution,
- a methodology (one or two paragraphs) covering the objective, the steps, the type of data mining model and why, the dataset source and size, pre-processing and cleaning, the algorithms, and how the model is validated.

We wrote all of this: a two-paragraph background and a two-paragraph methodology that describes the dataset, the cleaning, the five algorithms and the validation.

**Assignment 2 (Part 2)** asked us to add to the same document:
- a **Results and Discussions** section (one or two paragraphs) with the performance scores and whether the model met its objectives,
- a **Summary and Conclusion** paragraph with recommendations for future improvement,
- the **GitHub link**, placed just after the conclusion.

We wrote the Results section (two paragraphs, plus a model table and two figures), the Summary and Conclusion (one paragraph with future recommendations), and a GitHub Link section directly after it. Each member uploads their own copy of the document because grades are individual.

## 3. The dataset

We used the Kaggle dataset **"Student Habits vs Academic Performance"** *(add the exact dataset author and URL here)*. It is stored in `data/student_habits_performance.csv`.

- **Size:** 1,000 students and 16 columns.
- **Target (what we predict):** `exam_score`, from 18.4 to 100, with a mean of 69.6 and a standard deviation of 16.89.
- **Numeric factors:** `age`, `study_hours_per_day`, `social_media_hours`, `netflix_hours`, `attendance_percentage`, `sleep_hours`, `exercise_frequency`, `mental_health_rating`.
- **Category factors:** `gender`, `part_time_job`, `diet_quality`, `parental_education_level`, `internet_quality`, `extracurricular_participation`.
- **Identifier:** `student_id`, which only labels a student, so we do not use it for prediction.
- **Data quality:** there are no duplicate rows. The only missing values are 91 entries in `parental_education_level`.

## 4. What the code does, step by step

All of the analysis is in `student_performance_analysis.py`. We run it once and it does the following.

**Step 1: Load and clean.** It reads the CSV and checks for missing values and duplicates. We did not delete the 91 students with a missing parental education level, because that would throw away data and could bias the results. Instead we gave them their own category, "Not reported". We removed `student_id` because it only identifies a student.

**Step 2: Correlations.** For each numeric factor it calculates the Pearson correlation (r) and the Spearman correlation with exam score, plus a p-value. For attendance it does more:
- the correlation on its own,
- the partial correlation, which measures attendance after removing the effect of the other numeric factors,
- the average exam score for four attendance bands (below 70%, 70-79%, 80-89%, 90-100%),
- a t-test comparing students at 90% or above with students below 70%.

It also works out the average exam score for each category factor (for example, by diet quality).

**Step 3: Prepare the data for the models.** Models only understand numbers, so we use *one-hot encoding* to turn words such as "Male" and "Female" into 0 and 1 columns. We *scale* the numeric columns so they are on a similar range. Both steps sit inside a pipeline, so they are learned from the training students only.

**Step 4: Split.** We keep 80% of students (800) to train the models and set aside 20% (200) that the models never see until the final test. A fixed random seed (42) means we get the same split every time.

**Step 5: Train five models.** We compared:

| Model | Plain explanation |
|---|---|
| Linear Regression | Finds the best straight-line rule that adds up the effect of each factor |
| Ridge Regression | The same, with a small penalty that stops any single number growing too large |
| Decision Tree (depth 5) | Asks a chain of yes/no questions, such as "study hours above 4?" |
| Random Forest (300 trees) | Many decision trees that vote together |
| Gradient Boosting (200 trees) | Trees built one after another, each fixing the mistakes of the last |

**Step 6: Validate.** We use two checks.
- *Test set:* we score each model on the 200 unseen students.
- *10-fold cross-validation:* we split the 800 training students into 10 parts, train on 9 and test on 1, and repeat 10 times. If the scores stay close together, the result is stable and not luck.

We report three measures: **R²**, **RMSE** and **MAE** (explained in section 6).

**Step 7: Find what matters most.** We use *permutation importance*. We take the best model, shuffle one factor so it carries no information, and see how much the model's accuracy drops. A big drop means that factor matters. Then we refit the best model without attendance, to see exactly how much attendance adds.

**Step 8: Save everything.** The script writes the pictures (PNG files) and all results to `outputs/results.json`.

## 5. Results

**Correlation with exam score**

| Factor | Pearson r |
|---|---|
| Study hours per day | 0.825 |
| Mental health rating | 0.322 |
| Exercise frequency | 0.160 |
| Sleep hours | 0.122 |
| Attendance percentage | 0.090 |
| Social media hours | -0.167 |
| Netflix hours | -0.172 |
| Age | -0.009 |

**Attendance in detail**
- Attendance has a weak positive correlation with exam score (r = 0.09, p = 0.004). On its own it explains about 0.8% of the differences in scores.
- Average scores by attendance band: below 70% = 68.0, 70-79% = 68.2, 80-89% = 69.4, 90-100% = 71.5.
- The difference between the 90%+ group and the below-70% group is not significant at the 5% level (p = 0.071).
- After holding the other numeric factors equal, the partial correlation is 0.248, and each extra attendance point goes with about 0.14 more marks (about 1.4 marks per 10 points).
- Removing attendance from the best model only lowers test R² from 0.897 to 0.890.

**What affects exam scores most**
- Study hours: one more hour per day goes with about 9.6 more marks. Alone it explains about 68% of the variation (0.825 x 0.825).
- Mental health rating is next, at about 1.9 marks per rating point.
- More Netflix and social media time goes with lower scores (about 2.3 and 2.6 marks per extra hour).

**Model performance (200 unseen students)**

| Model | Test R² | RMSE | MAE | 10-fold CV R² |
|---|---|---|---|---|
| Linear Regression | 0.897 | 5.15 | 4.19 | 0.894 (+/- 0.022) |
| Ridge Regression | 0.897 | 5.15 | 4.19 | 0.894 (+/- 0.022) |
| Gradient Boosting | 0.880 | 5.55 | 4.62 | 0.878 (+/- 0.021) |
| Random Forest | 0.849 | 6.23 | 5.00 | 0.851 (+/- 0.024) |
| Decision Tree | 0.710 | 8.62 | 6.90 | 0.730 (+/- 0.034) |

**Did we meet our objectives?** Yes. We found out whether attendance matters (only weakly), identified what matters most (study hours first, then mental health), and built a model that predicts exam scores to within about five marks on average.

## 6. Terms explained in plain language

- **Linear regression:** a method that draws the best straight-line rule through the data. In our case the rule looks like: *exam score = a starting number + (9.6 x study hours) + (1.9 x mental health rating) - (2.6 x social media hours) + ... + (0.14 x attendance %)*. Each multiplier is a *coefficient*, the marks gained or lost when that factor goes up by one unit. The computer tries many values and keeps the ones that make the predicted scores closest to the real scores.
- **Correlation (r):** how closely two things move together, from -1 to +1. Near 0 means almost no link.
- **p-value:** the chance of seeing a result like this just by luck. Below 0.05 is usually called significant. With 1,000 students even a tiny effect can be significant, so we also look at how big the effect is.
- **R²:** the share of the differences in exam scores the model explains. Ours is 0.897, so about 90%.
- **RMSE and MAE:** the model's typical error, in exam marks. Ours is about five marks.
- **Train/test split and cross-validation:** testing on students the model has not seen, so we measure real predictive power and not memorising.
- **Permutation importance:** the drop in accuracy when a factor is shuffled. Study hours causes a drop above 1.0 because R² can go below zero when a key factor is destroyed.
- **Partial correlation:** the link between two things after removing the effect of other factors.
- **Association vs cause:** a link between two things does not prove one causes the other.

## 7. Files in this repository

| File or folder | What it is |
|---|---|
| `data/student_habits_performance.csv` | The dataset |
| `student_performance_analysis.py` | The full analysis (cleaning, correlations, models, importance, figures) |
| `build_dashboard.py` | Reads the CSV and `results.json` and builds the dashboard page |
| `dashboard_template.html` | The dashboard design, with a placeholder where the data goes in |
| `outputs/dashboard.html` | The finished dashboard |
| `index.html` | A copy of the dashboard so GitHub Pages shows it as the main page |
| `outputs/results.json` | Every number the analysis produced |
| `outputs/*.png` | Figures drawn by the script: `correlations`, `attendance_vs_score`, `study_vs_score`, `feature_importance`, `model_comparison` |

The PNG files and `results.json` are created by the script each time it runs, so deleting them and rerunning brings them back. The dashboard does not calculate anything itself. It only displays the numbers in `results.json`, so after changing the data we rerun both scripts.

## 8. The dashboard, page by page

- **Overview:** a summary of findings, four key figures (attendance r, study hours r, best model R², typical error), a bar chart of correlations and a bar chart of factor importance.
- **Dataset:** the number of records, columns, missing values and mean score, a histogram of exam scores, and a data dictionary listing all 16 columns with role, type, missing count and summary.
- **Attendance:** the correlation, partial correlation, marks per 10 attendance points and the t-test result, then a chart of average score by attendance band and a chart of model accuracy with and without attendance.
- **Drivers:** a factor explorer (choose a factor and see the average score for its four quartiles) and a chart of the effect of one extra unit of each factor.
- **Models:** a chart comparing the five models and a table of R², RMSE, MAE and cross-validation scores.

The charts load from the internet, so the dashboard needs a connection.

## 9. How to run it

1. Install Python 3.11 or newer and tick "Add to PATH" during setup.
2. Open a terminal in the project folder and install the libraries:
   `pip install pandas numpy scipy scikit-learn matplotlib`
3. Run the analysis (it takes a few minutes, because it trains five models many times):
   `python student_performance_analysis.py`
4. Build the dashboard:
   `python build_dashboard.py`
5. Open `outputs/dashboard.html` in a browser. To refresh the website copy, run `copy outputs\dashboard.html index.html`.

## 10. Publishing with GitHub Pages

After uploading the repository, we open **Settings**, then **Pages**, set **Source** to *Deploy from a branch*, choose branch `main` and folder `/ (root)`, and save. After a minute or two the dashboard is live at the link at the top of this file. The repository must be public on a free account.

## 11. Limitations

- The results show association, not cause. For example, students who study more score higher, but this data cannot prove that studying is the only reason.
- We do not know how the Kaggle data was collected, and habits such as study hours may be self-reported.
- It is one dataset of 1,000 students, so the findings may not apply everywhere, including to Kenyan universities.
- Linear regression working best suggests the relationships are mostly straight-line, which may not hold in other data.

## 12. Future improvements

- Test the model on real student records from Kenyan institutions.
- Add information such as past grades, the quality (not just the amount) of study time, and course type.
- Check whether the effect of attendance differs between courses.
- Try more advanced models with careful tuning.

## 13. References

- Korir, E. K. (2013). Students' school attendance and academic performance in Njoro District, Kenya: a relational analysis. Mount Kenya University.
- Kaggle. *Student Habits vs Academic Performance* [Dataset]. *(add the dataset author and URL)*

The libraries we used are pandas, NumPy, SciPy, scikit-learn and Matplotlib for the analysis, and Chart.js for the dashboard charts.
