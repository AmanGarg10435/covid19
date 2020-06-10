import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("ggplot")
data = pd.read_csv("owid-covid-data.csv")
data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))

covid_sg = data[data["iso_code"]=="SGP"]
covid_kr = data[data["iso_code"] == "KOR"]
covid_jp = data[data["iso_code"] == "JPN"]

covid_sg['log_total_cases'] = covid_sg['total_cases'].apply(lambda x: np.log(x) if x > 0 else x)
covid_sg['log_new_cases'] = covid_sg['new_cases'].apply(lambda x: np.log(x) if x > 0 else x)
covid_kr['log_total_cases'] = covid_kr['total_cases'].apply(lambda x: np.log(x) if x > 0 else x)
covid_kr['log_new_cases'] = covid_kr['new_cases'].apply(lambda x: np.log(x) if x > 0 else x)
covid_jp['log_total_cases'] = covid_jp['total_cases'].apply(lambda x: np.log(x) if x > 0 else x)
covid_jp['log_new_cases'] = covid_jp['new_cases'].apply(lambda x: np.log(x) if x > 0 else x)


cb_start = covid_sg.set_index('date').loc['2020-04-07'].total_cases
cb_extended = covid_sg.set_index('date').loc['2020-05-04'].total_cases
cb_end = covid_sg.set_index('date').loc['2020-06-01'].total_cases

cb_log_start = covid_sg.set_index('date').loc['2020-04-07'].log_total_cases
cb_log_extended = covid_sg.set_index('date').loc['2020-05-04'].log_total_cases
cb_log_end = covid_sg.set_index('date').loc['2020-06-01'].log_total_cases





# plt.plot(covid_sg["total_cases"],covid_sg["new_cases"],"r",label="SGP")
# plt.plot(covid_kr["total_cases"], covid_kr["new_cases"],"g",label="KOR")
# plt.plot(covid_jp["total_cases"], covid_jp["new_cases"],"b",label="JPN")

# plt.plot([cb_start, cb_start], [0, 1400], '-k',label="Start of Circuit Breaker")
# plt.plot([cb_extended, cb_extended], [0, 1400], '--k',label="Extension of Circuit Breaker")
# plt.plot([cb_end, cb_end], [0, 1400], ':k',label="End of Circuit Breaker")

# plt.xlabel("Total cases contracted")
# plt.ylabel("New cases per day")

# plt.legend()
# plt.show()



# plt.plot(covid_sg["log_total_cases"], covid_sg["log_new_cases"], "r", label="SGP")
# plt.plot(covid_kr["log_total_cases"], covid_kr["log_new_cases"], "g", label="KOR")
# plt.plot(covid_jp["log_total_cases"], covid_jp["log_new_cases"], "b", label="JPN")

# plt.plot([cb_log_start, cb_log_start], [0, 7], '-k',label="Start of Circuit Breaker")
# plt.plot([cb_log_extended, cb_log_extended], [0, 7],'--k', label="Extension of Circuit Breaker")
# plt.plot([cb_log_end, cb_log_end], [0, 7], ':k', label="End of Circuit Breaker")

# plt.xlabel("Total cases contracted (Log)")
# plt.ylabel("New cases per day (Log)")

# plt.legend()
# plt.show()



# Data prep for SGP

# Add week number
covid_sg['week_no'] = covid_sg['date'].apply(lambda x: x.week)

# Group by week number
covid_sg_groupby = covid_sg.groupby("week_no")

# Get last instances of by-groups
weekly_covid_sg = covid_sg_groupby.last()

# Calculate weekly averages (NOTE: SG does not have daily test rates per thousand, only a weekly rate)
weekly_average = []
for i in range(1, len(covid_sg_groupby.size())+1):
    weekly_average.append(
        round(covid_sg_groupby.get_group(i).mean().new_cases, 2))

# Create weekly average column
weekly_covid_sg['weekly_average'] = weekly_average

# log transform total cases and weekly average
weekly_covid_sg['log_total_cases'] = weekly_covid_sg['total_cases'].apply(
    lambda x: np.log(x) if x > 0 else x)
weekly_covid_sg['log_weekly_average'] = weekly_covid_sg['weekly_average'].apply(
    lambda x: np.log(x) if x > 0 else x)

# Data prep for KOR

# Add week number
covid_kr['week_no'] = covid_kr['date'].apply(lambda x: x.week)

# Group by week number
covid_kr_groupby = covid_kr.groupby("week_no")

# Get last instances of by-groups
weekly_covid_kr = covid_kr_groupby.last()

# Calculate weekly averages
weekly_average = []
weekly_average_tests = []  # calculate average weekly test rates for comparison with SG
for i in range(1, len(covid_kr_groupby.size())+1):
    weekly_average.append(round(covid_kr_groupby.get_group(i).mean().new_cases, 2))
    weekly_average_tests.append(
        round(covid_kr_groupby.get_group(i).mean().total_tests_per_thousand, 2))

# Create weekly average column
weekly_covid_kr['weekly_average'] = weekly_average
weekly_covid_kr['weekly_average_tests'] = weekly_average_tests

# log transform total cases and weekly average
weekly_covid_kr['log_total_cases'] = weekly_covid_kr['total_cases'].apply(
    lambda x: np.log(x) if x > 0 else x)
weekly_covid_kr['log_weekly_average'] = weekly_covid_kr['weekly_average'].apply(
    lambda x: np.log(x) if x > 0 else x)

# Data prep for JPN

# Add week number
covid_jp['week_no'] = covid_jp['date'].apply(lambda x: x.week)

# Group by week number
covid_jp_groupby = covid_jp.groupby("week_no")

# Get last instances of by-groups
weekly_covid_jp = covid_jp_groupby.last()

# Calculate weekly averages
weekly_average = []
weekly_average_tests = []  # calculate average weekly test rates for comparison with SG
for i in range(1, len(covid_jp_groupby.size())+1):
    weekly_average.append(
        round(covid_jp_groupby.get_group(i).mean().new_cases, 2))
    weekly_average_tests.append(
        round(covid_jp_groupby.get_group(i).mean().total_tests_per_thousand, 2))

# Create weekly average column
weekly_covid_jp['weekly_average'] = weekly_average
weekly_covid_jp['weekly_average_tests'] = weekly_average_tests

# log transform total cases and weekly average
weekly_covid_jp['log_total_cases'] = weekly_covid_jp['total_cases'].apply(
    lambda x: np.log(x) if x > 0 else x)
weekly_covid_jp['log_weekly_average'] = weekly_covid_jp['weekly_average'].apply(
    lambda x: np.log(x) if x > 0 else x)


# plt.plot(weekly_covid_sg["total_cases"],weekly_covid_sg["weekly_average"],"r",label="SGP")
# plt.plot(weekly_covid_kr["total_cases"],weekly_covid_kr["weekly_average"], "g", label="KOR")
# plt.plot(weekly_covid_jp["total_cases"],weekly_covid_jp["weekly_average"], "b", label="JPN")

# plt.plot([cb_start, cb_start], [0, 1400], '-k',label="Start of Circuit Breaker")
# plt.plot([cb_extended, cb_extended], [0, 1400],'--k', label="Extension of Circuit Breaker")
# plt.plot([cb_end, cb_end], [0, 1400], ':k', label="End of Circuit Breaker")

# plt.xlabel("Total cases contracted")
# plt.ylabel("New cases per week")
# plt.legend()
# plt.show()


# plt.plot(weekly_covid_sg["log_total_cases"],weekly_covid_sg["log_weekly_average"], "r", label="SGP")
# plt.plot(weekly_covid_kr["log_total_cases"],
#          weekly_covid_kr["log_weekly_average"], "g", label="KOR")
# plt.plot(weekly_covid_jp["log_total_cases"],
#          weekly_covid_jp["log_weekly_average"], "b", label="JPN")

# plt.plot([cb_log_start, cb_log_start], [0, 7], '-k',
#          label="Start of Circuit Breaker")
# plt.plot([cb_log_extended, cb_log_extended], [0, 7],
#          '--k', label="Extension of Circuit Breaker")
# plt.plot([cb_log_end, cb_log_end], [0, 7], ':k', label="End of Circuit Breaker")

# plt.xlabel("Total cases contracted (log)")
# plt.ylabel("New cases per week (log)")
# plt.legend()
# plt.show()



weekly_covid_sg.loc[20, 'total_tests_per_thousand'] = weekly_covid_sg.loc[19,'total_tests_per_thousand']
# plt.plot(weekly_covid_sg['total_cases'], weekly_covid_sg['total_tests_per_thousand'],'r', label="SGP")
# plt.plot(weekly_covid_kr['total_cases'], weekly_covid_kr['weekly_average_tests'],'g', label="KOR")
# plt.plot(weekly_covid_jp['total_cases'], weekly_covid_jp['weekly_average_tests'],'b', label="JPN")


# plt.plot([cb_start, cb_start], [0, 40], '-k',
#           label="Start of Circuit Breaker")
# plt.annotate("Start - 7 Apr", (cb_start, 0),
#               textcoords="offset points", xytext=(-20, 400))
# plt.plot([cb_extended, cb_extended], [0, 40],
#           '--k', label="Extended Circuit Breaker")
# plt.annotate("Extended - 4 May", (cb_extended, 0),
#               textcoords="offset points", xytext=(-20, 400))
# plt.plot([cb_end, cb_end], [0, 40], ':k', label="End of Circuit Breaker")
# plt.annotate("End - 2 Jun", (cb_end, 0),
#               textcoords="offset points", xytext=(-20, 400))

# plt.xlabel("Total cases per week")
# plt.ylabel("Total tests per thousand people")
# plt.legend()
# plt.show()

x = np.array(weekly_covid_sg["log_total_cases"]).reshape(-1,1)
y = np.array(weekly_covid_sg["log_weekly_average"]).reshape(-1,1)

linear = linear_model.LinearRegression()
linear.fit(x,y)

plt.plot(x,y,"r")
plt.plot(x,linear.predict(x),"b")
plt.show()
