{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Dataset load"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from xgboost import XGBRegressor\n",
    "from xgboost import plot_importance\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "from sklearn.metrics import r2_score, mean_squared_error\n",
    "from sklearn.model_selection import GridSearchCV, train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from scipy import stats\n",
    "from scipy.stats import norm, skew\n",
    "\n",
    "# train / test set load\n",
    "train = pd.read_csv(\"C:/ITWILL/4_Python-II/data/FIFA_train.csv\")\n",
    "test = pd.read_csv(\"C:/ITWILL/4_Python-II/data/FIFA_test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 8932 entries, 0 to 8931\n",
      "Data columns (total 12 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   id                8932 non-null   int64  \n",
      " 1   name              8932 non-null   object \n",
      " 2   age               8932 non-null   int64  \n",
      " 3   continent         8932 non-null   object \n",
      " 4   contract_until    8932 non-null   object \n",
      " 5   position          8932 non-null   object \n",
      " 6   prefer_foot       8932 non-null   object \n",
      " 7   reputation        8932 non-null   float64\n",
      " 8   stat_overall      8932 non-null   int64  \n",
      " 9   stat_potential    8932 non-null   int64  \n",
      " 10  stat_skill_moves  8932 non-null   float64\n",
      " 11  value             8932 non-null   float64\n",
      "dtypes: float64(3), int64(4), object(5)\n",
      "memory usage: 837.5+ KB\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 3828 entries, 0 to 3827\n",
      "Data columns (total 11 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   id                3828 non-null   int64  \n",
      " 1   name              3828 non-null   object \n",
      " 2   age               3828 non-null   int64  \n",
      " 3   continent         3828 non-null   object \n",
      " 4   contract_until    3828 non-null   object \n",
      " 5   position          3828 non-null   object \n",
      " 6   prefer_foot       3828 non-null   object \n",
      " 7   reputation        3828 non-null   float64\n",
      " 8   stat_overall      3828 non-null   int64  \n",
      " 9   stat_potential    3828 non-null   int64  \n",
      " 10  stat_skill_moves  3828 non-null   float64\n",
      "dtypes: float64(2), int64(4), object(5)\n",
      "memory usage: 329.1+ KB\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>age</th>\n",
       "      <th>reputation</th>\n",
       "      <th>stat_overall</th>\n",
       "      <th>stat_potential</th>\n",
       "      <th>stat_skill_moves</th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8932.000000</td>\n",
       "      <td>8.932000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>7966.775750</td>\n",
       "      <td>25.209136</td>\n",
       "      <td>1.130878</td>\n",
       "      <td>67.091133</td>\n",
       "      <td>71.997201</td>\n",
       "      <td>2.401702</td>\n",
       "      <td>2.778673e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>4844.428521</td>\n",
       "      <td>4.635515</td>\n",
       "      <td>0.423792</td>\n",
       "      <td>6.854910</td>\n",
       "      <td>5.988147</td>\n",
       "      <td>0.776048</td>\n",
       "      <td>5.840982e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>47.000000</td>\n",
       "      <td>48.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3751.750000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>63.000000</td>\n",
       "      <td>68.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.750000e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>7696.500000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>67.000000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>8.250000e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>12082.250000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.600000e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>16948.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1.105000e+08</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 id          age   reputation  stat_overall  stat_potential  \\\n",
       "count   8932.000000  8932.000000  8932.000000   8932.000000     8932.000000   \n",
       "mean    7966.775750    25.209136     1.130878     67.091133       71.997201   \n",
       "std     4844.428521     4.635515     0.423792      6.854910        5.988147   \n",
       "min        0.000000    16.000000     1.000000     47.000000       48.000000   \n",
       "25%     3751.750000    21.000000     1.000000     63.000000       68.000000   \n",
       "50%     7696.500000    25.000000     1.000000     67.000000       72.000000   \n",
       "75%    12082.250000    28.000000     1.000000     72.000000       76.000000   \n",
       "max    16948.000000    40.000000     5.000000     94.000000       94.000000   \n",
       "\n",
       "       stat_skill_moves         value  \n",
       "count       8932.000000  8.932000e+03  \n",
       "mean           2.401702  2.778673e+06  \n",
       "std            0.776048  5.840982e+06  \n",
       "min            1.000000  1.000000e+04  \n",
       "25%            2.000000  3.750000e+05  \n",
       "50%            2.000000  8.250000e+05  \n",
       "75%            3.000000  2.600000e+06  \n",
       "max            5.000000  1.105000e+08  "
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dataset info\n",
    "train.info() # (8932, 12)\n",
    "test.info()  # (3828, 11)\n",
    "train.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       4.0\n",
       "1       1.0\n",
       "2       3.0\n",
       "3       3.0\n",
       "4       1.0\n",
       "       ... \n",
       "8927    3.0\n",
       "8928    2.0\n",
       "8929    2.0\n",
       "8930    1.0\n",
       "8931    2.0\n",
       "Name: stat_skill_moves, Length: 8932, dtype: float64"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dataset check\n",
    "train.head()\n",
    "\n",
    "# 연속형 변수 / 범주형변수 분리\n",
    "'''\n",
    "연속형 변수 : 'age', , 'stat_overall', 'stat_potential'\n",
    "범주형 변수 : 'continent', 'contract_until', 'position', 'prefer_foot', 'reputation'\n",
    "'id', 'name', 'value', stat_skill_moves\n",
    "'''\n",
    "train.columns\n",
    "\n",
    "train[\"stat_skill_moves\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>contract_until</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2018</th>\n",
       "      <td>327</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2019</th>\n",
       "      <td>2366</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020</th>\n",
       "      <td>2041</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021</th>\n",
       "      <td>2308</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2022</th>\n",
       "      <td>761</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023</th>\n",
       "      <td>506</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2024</th>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025</th>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2026</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Dec 31, 2018</th>\n",
       "      <td>64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jan 1, 2019</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jan 12, 2019</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jan 31, 2019</th>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jun 30, 2019</th>\n",
       "      <td>501</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jun 30, 2020</th>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>May 31, 2019</th>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>May 31, 2020</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                value\n",
       "contract_until       \n",
       "2018              327\n",
       "2019             2366\n",
       "2020             2041\n",
       "2021             2308\n",
       "2022              761\n",
       "2023              506\n",
       "2024               12\n",
       "2025                3\n",
       "2026                1\n",
       "Dec 31, 2018       64\n",
       "Jan 1, 2019         2\n",
       "Jan 12, 2019        1\n",
       "Jan 31, 2019       10\n",
       "Jun 30, 2019      501\n",
       "Jun 30, 2020        9\n",
       "May 31, 2019       19\n",
       "May 31, 2020        1"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import re\n",
    "train[[\"contract_until\", \"value\"]].groupby(\"contract_until\").count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "comdata = pd.concat([train, test], ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py:9: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  if __name__ == '__main__':\n"
     ]
    }
   ],
   "source": [
    "# contract_until 변수 전처리(월, 일 제거)\n",
    "idx = []\n",
    "for i, value in enumerate(comdata[\"contract_until\"]):\n",
    "    if re.match(\"[A-Z]{1,}\", value):\n",
    "        idx.append(i)\n",
    "comdata[\"contract_until\"][idx].replace(\"[A-Z]{1,}[a-z]{2}\", \"\")\n",
    "\n",
    "for i, j in zip(comdata[\"contract_until\"][idx], idx):\n",
    "    comdata[\"contract_until\"][j] = re.sub(\"[A-Z]{1,}[a-z]{2} [0-9]{1,}, \", \"\", i).strip()\n",
    "    \n",
    "# 공백 제거\n",
    "comdata.loc[:, \"contract_until\"] = comdata.loc[:, \"contract_until\"].str.strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2572d2ecb08>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# contract_until \n",
    "# 변수에 따라 선수가치가 변하므로 사용!\n",
    "sns.barplot(x = \"contract_until\", y = \"value\", data = comdata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>continent</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>africa</th>\n",
       "      <td>2.972247e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>asia</th>\n",
       "      <td>1.035146e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>europe</th>\n",
       "      <td>2.928125e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>oceania</th>\n",
       "      <td>8.225429e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>south america</th>\n",
       "      <td>3.183204e+06</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      value\n",
       "continent                  \n",
       "africa         2.972247e+06\n",
       "asia           1.035146e+06\n",
       "europe         2.928125e+06\n",
       "oceania        8.225429e+05\n",
       "south america  3.183204e+06"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZ4AAAEGCAYAAABVSfMhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAerElEQVR4nO3de5hU1Znv8e8PEO+KSHsJkOBJSCJxItEeZMZM4qgH0eck4Byd0TMJaDgh8WASk0wIyZkTjcZn1Fw8j7mYISMBPMkQo8mIBkUGb7mptIpcNIYOGmkRQVHjJVGB9/yxV4dNW9Xd1VCruunf53nqqV3vXnutVbu66+29atVqRQRmZma5DGh0B8zMrH9x4jEzs6yceMzMLCsnHjMzy8qJx8zMshrU6A70dsOGDYtRo0Y1uhtmZn3K/fff/0xENFXa58TThVGjRtHS0tLobpiZ9SmSfl9tn4fazMwsKyceMzPLyonHzMyycuIxM7Os6pZ4JO0l6T5JD0laLenLKT5X0mOSlqfb2BSXpKsktUpaIemYUl1TJa1Jt6ml+LGSVqZjrpKkFB8qaUkqv0TSQV21YWZmedTziudV4MSIOBoYC0yUND7t+1xEjE235Sl2KjA63aYDV0ORRIALgeOAccCF7YkklZleOm5iis8ClkbEaGBpely1DTMzy6duiScKL6WHe6RbZ0thTwLmp+PuAYZIOhw4BVgSEZsj4jlgCUUSOxw4ICJ+HcUS2/OByaW65qXteR3ildowM7NM6voZj6SBkpYDGymSx71p16VpqOtKSXum2HBgXenwthTrLN5WIQ5waEQ8BZDuD+mijY79ni6pRVLLpk2banrOZmbWubomnojYGhFjgRHAOElHAV8A3gn8JTAU+HwqrkpV9CDemW4dExGzI6I5Ipqbmip+8TarmTNnMmXKFGbOnNnorpiZ7bQss9oi4nngTmBiRDyVhrpeBb5P8bkNFFcfI0uHjQDWdxEfUSEO8HT7EFq639hFG73ahg0bePLJJ9mwYUOju2JmttPqOautSdKQtL03cDLwm1JCEMVnL6vSIQuBKWnm2XjghTRMthiYIOmgNKlgArA47XtR0vhU1xTgxlJd7bPfpnaIV2rDzMwyqedabYcD8yQNpEhw10XEzZJul9REMey1HPh4Kr8IOA1oBV4BzgWIiM2SLgGWpXIXR8TmtH0eMBfYG7gl3QAuA66TNA14AjizszbM+qKZM2eyYcMGDjvsMK644opGd8es2+qWeCJiBfCeCvETq5QPYEaVfXOAORXiLcBRFeLPAifV0oZZX9M+BGvW13jlAjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLyonHzMyycuIxM7OsnHjMzCwrJx4zM8vKicfMzLJy4jEzs6yceMzMLCsnHjMzy6qe/xZht3fs5+ZnaWf/Z15kIPDEMy/Wvc37vzqlrvWbmfmKx8zMsnLiMTOzrDzUZn2K/+umWd/nxGN9iv/rplnf56E2MzPLyonHzMyycuIxM7Os6pZ4JO0l6T5JD0laLenLKX6EpHslrZH0I0mDU3zP9Lg17R9VqusLKf6opFNK8Ykp1ippVilecxtmZpZHPa94XgVOjIijgbHAREnjgcuBKyNiNPAcMC2VnwY8FxFvA65M5ZA0BjgLeBcwEfiOpIGSBgLfBk4FxgBnp7LU2oaZmeVTt8QThZfSwz3SLYATgetTfB4wOW1PSo9J+0+SpBRfEBGvRsRjQCswLt1aI2JtRLwGLAAmpWNqbcPMzDKp62c86cpkObARWAL8Dng+IrakIm3A8LQ9HFgHkPa/ABxcjnc4plr84B600bHf0yW1SGrZtGlTz568mZlVVNfEExFbI2IsMILiCuXISsXSfaUrj9iF8c7a2DEQMTsimiOiuampqcIhZmbWU1m+QBoRz0u6ExgPDJE0KF1xjADWp2JtwEigTdIg4EBgcynernxMpfgzPWjDbJc5/pvHZ2ln8PODGcAA1j2/ru5t/vITv6xr/da/1HNWW5OkIWl7b+Bk4BHgDuCMVGwqcGPaXpgek/bfHhGR4melGWlHAKOB+4BlwOg0g20wxQSEhemYWtswM7NM6nnFczgwL80+GwBcFxE3S3oYWCDpK8CDwDWp/DXAtZJaKa5CzgKIiNWSrgMeBrYAMyJiK4Ck84HFwEBgTkSsTnV9vpY2bOc9cfFfZGlny+ahwCC2bP593dt885dW1rV+s/6qboknIlYA76kQX0vxeU/H+J+AM6vUdSlwaYX4ImDRrmjDzMzy8MoFZmaWlROPmZll5X+L0AdsG7zvDvdmZn2ZE08f8PLoCY3ugpnZLuOhNjMzy8qJx8zMsnLiMTOzrJx4zMwsK08usD5l2F7bgC3p3sz6Iice61P+6d3PN7oLZraTPNRmZmZZOfGYmVlWTjxmZpaVE4+ZmWXlxGNmZlk58ZiZWVZOPGZmlpUTj5mZZeXEY2ZmWXnlArM+KvYJtrGN2Cca3RWzmjjxmPVRrx//eqO7YNYjdRtqkzRS0h2SHpG0WtKnUvwiSU9KWp5up5WO+YKkVkmPSjqlFJ+YYq2SZpXiR0i6V9IaST+SNDjF90yPW9P+UV21YWZmedTzM54twGcj4khgPDBD0pi078qIGJtuiwDSvrOAdwETge9IGihpIPBt4FRgDHB2qZ7LU12jgeeAaSk+DXguIt4GXJnKVW2jfqfAzMw6qlviiYinIuKBtP0i8AgwvJNDJgELIuLViHgMaAXGpVtrRKyNiNeABcAkSQJOBK5Px88DJpfqmpe2rwdOSuWrtWFmZplkmdWWhrreA9ybQudLWiFpjqSDUmw4sK50WFuKVYsfDDwfEVs6xHeoK+1/IZWvVlfH/k6X1CKpZdOmTTU/XzMzq67uiUfSfsANwAUR8QfgauCtwFjgKeDr7UUrHB49iPekrh0DEbMjojkimpuamiocYmZmPVXXxCNpD4qk84OI+AlARDwdEVsjYhvwPbYPdbUBI0uHjwDWdxJ/BhgiaVCH+A51pf0HAps7qcvMzDKp56w2AdcAj0TEN0rxw0vFTgdWpe2FwFlpRtoRwGjgPmAZMDrNYBtMMTlgYUQEcAdwRjp+KnBjqa6pafsM4PZUvlobZmaWST2/x3M88GFgpaTlKfZFillpYymGuB4HPgYQEaslXQc8TDEjbkZEbAWQdD6wGBgIzImI1am+zwMLJH0FeJAi0ZHur5XUSnGlc1ZXbZiZWR51SzwR8Qsqf6ayqJNjLgUurRBfVOm4iFhLhVlpEfEn4Mxa2jAzszy8VpuZmWXlxGNmZlk58ZiZWVZOPGZmlpUTj5mZZeXEY2ZmWTnxmJlZVk48ZmaWlROPmZll5cRjZmZZOfGYmVlWTjxmZpaVE4+ZmWXlxGNmZlk58ZiZWVZOPGZmlpUTj5mZZeXEY2ZmWTnxmJlZVk48ZmaWVd0Sj6SRku6Q9Iik1ZI+leJDJS2RtCbdH5TiknSVpFZJKyQdU6praiq/RtLUUvxYSSvTMVdJUk/bMDOzPOp5xbMF+GxEHAmMB2ZIGgPMApZGxGhgaXoMcCowOt2mA1dDkUSAC4HjgHHAhe2JJJWZXjpuYorX1IaZmeVTt8QTEU9FxANp+0XgEWA4MAmYl4rNAyan7UnA/CjcAwyRdDhwCrAkIjZHxHPAEmBi2ndARPw6IgKY36GuWtowM7NMsnzGI2kU8B7gXuDQiHgKiuQEHJKKDQfWlQ5rS7HO4m0V4vSgDTMzy6TLxCPpUEnXSLolPR4jaVp3G5C0H3ADcEFE/KGzohVi0YN4p93pzjGSpktqkdSyadOmLqo0M7NadOeKZy6wGHhTevxb4ILuVC5pD4qk84OI+EkKP90+vJXuN6Z4GzCydPgIYH0X8REV4j1pYwcRMTsimiOiuampqTtP1czMuqk7iWdYRFwHbAOIiC3A1q4OSjPMrgEeiYhvlHYtBNpnpk0FbizFp6SZZ+OBF9Iw2WJggqSD0qSCCcDitO9FSeNTW1M61FVLG2ZmlsmgbpR5WdLBpCGp9jfsbhx3PPBhYKWk5Sn2ReAy4Lo0XPcEcGbatwg4DWgFXgHOBYiIzZIuAZalchdHxOa0fR7FFdnewC3pRq1tmJlZPt1JPJ+huFJ4q6RfAk3AGV0dFBG/oPJnKgAnVSgfwIwqdc0B5lSItwBHVYg/W2sbZmaWR5eJJyIekPR+4B0UieTRiHi97j0zM7PdUpeJR9KUDqFjJBER8+vUJzMz2411Z6jtL0vbe1EMYT1A8YVNMzOzmnRnqO0T5ceSDgSurVuPzMxst9aTlQteoVjrzMzMrGbd+YznJrZ/u38AMAa4rp6dMjOz3Vd3PuP5Wml7C/D7iGirVtjMzKwz3fmM564cHTEzs/6hauKR9CKVF90UxXcxD6hbr8zMbLdVNfFExP45O2JmZv1Ddz7jAUDSIRTf4wEgIp6oS4/MzGy31p3/x/NBSWuAx4C7gMfZvhinmZlZTbrzPZ5LgPHAbyPiCIqVC35Z116ZmdluqzuJ5/W02vMASQMi4g5gbJ37ZWZmu6nufMbzfPr31T8HfiBpI8X3eczMzGrWnSueu4EhwKeAW4HfAR+oZ6fMzGz31Z3EI4p/P30nsB/wozT0ZmZmVrMuE09EfDki3kXxnzvfBNwl6T/r3jMzM9st1bI69UZgA/AscEh9umNmZru77nyP5zxJdwJLgWHARyPi3fXumJmZ7Z66M6vtLcAFEbG83p0xM7PdX3c+45nVk6QjaY6kjZJWlWIXSXpS0vJ0O6207wuSWiU9KumUUnxiirVKmlWKHyHpXklrJP1I0uAU3zM9bk37R3XVhpmZ5dOT/0DaXXOBiRXiV0bE2HRbBCBpDHAW8K50zHckDZQ0EPg2cCrFP6A7O5UFuDzVNRp4DpiW4tOA5yLibcCVqVzVNnbxczYzsy7ULfFExN3A5m4WnwQsiIhXI+IxoBUYl26tEbE2Il4DFgCTJAk4Ebg+HT8PmFyqa17avh44KZWv1oaZmWVUzyueas6XtCINxR2UYsOBdaUybSlWLX4w8HxEbOkQ36GutP+FVL5aXW8gabqkFkktmzZt6tmzNDOzinInnquBt1Ks9fYU8PUUV4Wy0YN4T+p6YzBidkQ0R0RzU1NTpSJmZtZDWRNPRDwdEVsjYhvwPbYPdbUBI0tFRwDrO4k/AwyRNKhDfIe60v4DKYb8qtVlZmYZZU08kg4vPTwdaJ/xthA4K81IOwIYDdwHLANGpxlsgykmByyMiADuAM5Ix08FbizVNTVtnwHcnspXa8PMzDLq9n8grZWkfwdOAIZJagMuBE6QNJZiiOtx4GMAEbFa0nXAwxQrX8+IiK2pnvMp1oobCMyJiNWpic8DCyR9BXgQuCbFrwGuldRKcaVzVldtmJlZPnVLPBFxdoXwNRVi7eUvBS6tEF8ELKoQX0uFWWkR8SfgzFraMDOzfBoxq83MzPoxJx4zM8vKicfMzLJy4jEzs6yceMzMLCsnHjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLyonHzMyycuIxM7Os6rZIqJlZLjNnzmTDhg0cdthhXHHFFY3ujnXBicfM+rwNGzbw5JNPNrob1k0eajMzs6yceMzMLCsnHjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLqm6JR9IcSRslrSrFhkpaImlNuj8oxSXpKkmtklZIOqZ0zNRUfo2kqaX4sZJWpmOukqSetmFmtruYOXMmU6ZMYebMmY3uSlX1/ALpXOBbwPxSbBawNCIukzQrPf48cCowOt2OA64GjpM0FLgQaAYCuF/Swoh4LpWZDtwDLAImArfU2kbdnr2Zcdf73p+lnT8OGggSf2xrq3ub77/7rrrWv7P6wpdp63bFExF3A5s7hCcB89L2PGByKT4/CvcAQyQdDpwCLImIzSnZLAEmpn0HRMSvIyIoktvkHrZhZmYZ5f6M59CIeAog3R+S4sOBdaVybSnWWbytQrwnbbyBpOmSWiS1bNq0qaYnaGZmnestkwtUIRY9iPekjTcGI2ZHRHNENDc1NXVRrZmZ1SJ34nm6fXgr3W9M8TZgZKncCGB9F/ERFeI9acPMzDLKnXgWAu0z06YCN5biU9LMs/HAC2mYbDEwQdJBaXbaBGBx2veipPFpNtuUDnXV0oaZmWVUt1ltkv4dOAEYJqmNYnbaZcB1kqYBTwBnpuKLgNOAVuAV4FyAiNgs6RJgWSp3cUS0T1g4j2Lm3N4Us9luSfGa2jAzs7zqlngi4uwqu06qUDaAGVXqmQPMqRBvAY6qEH+21jbMzCyf3jK5wMzM+gn/B1Iz6/OGROxwb72bE4+Z9Xkf2rqt0V2wGniozczMsnLiMTOzrDzUZmaWwbc+e1OWdp5/5uU/39e7zfO//oEeHecrHjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLyonHzMyycuIxM7OsnHjMzCwrJx4zM8vKicfMzLJy4jEzs6y8VpuZ2W5k38EH7HDfGznxmJntRo5/6981ugtd8lCbmZll5cRjZmZZNSTxSHpc0kpJyyW1pNhQSUskrUn3B6W4JF0lqVXSCknHlOqZmsqvkTS1FD821d+ajlVnbZiZWT6NvOL524gYGxHN6fEsYGlEjAaWpscApwKj0206cDUUSQS4EDgOGAdcWEokV6ey7cdN7KINMzPLpDcNtU0C5qXtecDkUnx+FO4Bhkg6HDgFWBIRmyPiOWAJMDHtOyAifh0RAczvUFelNszMLJNGJZ4AbpN0v6TpKXZoRDwFkO4PSfHhwLrSsW0p1lm8rUK8szZ2IGm6pBZJLZs2berhUzQzs0oaNZ36+IhYL+kQYImk33RSVhVi0YN4t0XEbGA2QHNzc03HmplZ5xpyxRMR69P9RuCnFJ/RPJ2GyUj3G1PxNmBk6fARwPou4iMqxOmkDTMzyyR74pG0r6T927eBCcAqYCHQPjNtKnBj2l4ITEmz28YDL6RhssXABEkHpUkFE4DFad+Lksan2WxTOtRVqQ0zM8ukEUNthwI/TTOcBwE/jIhbJS0DrpM0DXgCODOVXwScBrQCrwDnAkTEZkmXAMtSuYsjYnPaPg+YC+wN3JJuAJdVacPMzDLJnngiYi1wdIX4s8BJFeIBzKhS1xxgToV4C3BUd9swM7N8etN0ajMz6weceMzMLCsnHjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLyonHzMyycuIxM7OsnHjMzCwrJx4zM8vKicfMzLJy4jEzs6yceMzMLCsnHjMzy8qJx8zMsnLiMTOzrJx4zMwsKyceMzPLyonHzMyycuIxM7Os+mXikTRR0qOSWiXNanR/zMz6k36XeCQNBL4NnAqMAc6WNKaxvTIz6z/6XeIBxgGtEbE2Il4DFgCTGtwnM7N+QxHR6D5kJekMYGJE/M/0+MPAcRFxfqnMdGB6evgO4NHsHX2jYcAzje5EL+FzsZ3PxXY+F9v1hnPxlohoqrRjUO6e9AKqENsh+0bEbGB2nu50j6SWiGhudD96A5+L7XwutvO52K63n4v+ONTWBowsPR4BrG9QX8zM+p3+mHiWAaMlHSFpMHAWsLDBfTIz6zf63VBbRGyRdD6wGBgIzImI1Q3uVnf0qqG/BvO52M7nYjufi+169bnod5MLzMyssfrjUJuZmTWQE4+ZmWXlxFNnkk6Q9Nelx3PTd4l6BUm/anQfrDpJZ0p6RNIdFfa9SdL1jehXbyXpYkknN7offYWkRZKG5G63300uaIATgJeAXvUGL2lgRGyNiL/uunTvIGlQRGxpdD8ymwb8r4jYIfGkc7Ee6DV/xPQGEfGlRvehL4mI0xrRrq94qpC0r6SfSXpI0ipJ/5DiJ0l6UNJKSXMk7Znij0salrabJd0paRTwceDTkpZL+ptU/fsk/UrS2mpXP5L+Q9L9klanlRTa4y9Jujzt+09J41JbayV9MJUZKOmrkpZJWiHpYyl+gqQ7JP0QWNleX6numel5PSTpshT7aKrnIUk3SNpnF5zbD0m6L52Tf039LffjDElz0/ZcSd9If/FfLmloOjcrJN0j6d2p3EWSrpV0u6Q1kj5aqu9zpXPx5Z3tf710fM0lfQl4L/Dd9HqeI+nHkm4CbpM0StKqdOxASV9Lr98KSZ9I8S+l575K0mxJlb5A3etVODcD08/GqvScP53K/XlEoa89d0mfSX1dJemCFJuSXs+HJF2bYk3pd3FZuh2f4uPS+8qD6f4dKX6OpJ9IujX9blxRarP8vlXxPacuIsK3CjfgvwPfKz0+ENgLWAe8PcXmAxek7ceBYWm7GbgzbV8E/FOpnrnAjymS/hiKdeMqtT803e8NrAIOTo8DODVt/xS4DdgDOBpYnuLTgX9O23sCLcARFFdfLwNHlNp5Kd2fSnFVtk+H9g8ulf0K8ImdPK9HAjcBe6TH3wGmtPcjxc4A5pbO183AwPT4m8CFafvE0nO+CHgona9h6XV6EzCBYmqp0jm/GXhfo3++uvuaA3cCzSl+DsUXoNvLjQJWpe3zgBuAQR3qGlqq/1rgA41+nrvo3BwLLCntH1L6eTmjrz339HxWAvsC+wGrgeMplutqf19pPwc/BN6btt8MPJK2Dyi9/icDN5R+btay/T3s98DItO/xCvXv8J5Tj5uH2qpbCXxN0uXAzRHxc0lHA49FxG9TmXnADOD/1lj3f0TENuBhSYdWKfNJSaen7ZHAaOBZ4DXg1lIfX42I1yWtpHgjguLN9t3afjV1YDr+NeC+iHisQnsnA9+PiFcAImJzih8l6SvAEIpfiMU1PteOTqL4JVuW/gDdG9jYxTE/joitafu9FH8UEBG3SzpY0oFp340R8Ufgj+kKaVwqPwF4MJXZj+Jc3L2Tz6MeKr3mHS0pvTZlJwPfjTQUWSrzt5JmAvsAQyne0G7atd3OouO5GQz8F0nfBH5G8QdYR33pub8X+GlEvAwg6ScUf8BeHxHPwA6v6cnAmNIF3AGS9qf4PZ8naTTFH6h7lOpfGhEvpLofBt5C8cdZWbX3nF3OiaeKiPitpGOB04B/kXQbna9wsIXtQ5d7dVH9q6XtN1z+SzqB4ofrryLiFUl3lup8PdKfJcC29roiYpuk9tdTFFcmOySJVO/LVfokOqxZl8wFJkfEQ5LOobhq2hkC5kXEFzr07bOlhx3PX7nPna2117H/kcr/S0T8aw/6mk0Xr3lZt18/SXtRXFE2R8Q6SRdVqbNXq3Ju9qS4yj+F4o+/vwc+Ujqmrz33aj/XlX4nB1Cciz/uUEGRhO+IiNNVDPPfWdpdfs/ZSof3/hp+/nYJf8ZThaQ3Aa9ExP8DvgYcA/wGGCXpbanYh4G70vbjFH/JQ/qLPHkR2L/G5g8Enks/AO8Extd4/GLgPEl7pOfydkn7dnHMbcBHlD7DkTQ0xfcHnkp1/WON/ahkKXCGpEPa25H0FuBpSUdKGgCc3snxd7f3I/2yPBMRf0j7JknaS9LBFAlyGcW5+Iik/dIxw9vb7mV29jW/Dfh4+x8f6fVrf+N4Jj3/vjoRodK5GQYMiIgbgP9D8ftZ1tee+93AZEn7pN/V04H7gb9PP8/l38nbgPJq+mPT5oHAk2n7nBrb39mfv5r4iqe6vwC+Kmkb8DpwXkT8SdK5wI/TL/gy4Lup/JeBayR9Ebi3VM9NwPWSJgGf6Gbbt1K8iaygGOO9p8a+/xvFsNsD6QPVTcDkzg6IiFvTD3CLpNeARcAXKX6p76UYF15J7Um0YzsPS/pnig/HB1Cc2xnALIrPX9ZRjC/vV6WKi4Dvp3PzCjC1tO8+imGXNwOXRDHra72kI4Ffp6GJl4AP0fXwXm674jV/O7BC0usUn09+S9L3KF63xyl+XvuiSudmOHBn+hkC2OEKOiKe70vPPSIeUDGh5r4U+reI+KWkS4G7JG2lGC4+B/gk8O10PgZRJK2PA1dQDLV9Bri9xi7s7M9fTbxkju0W0lDKSxHxtUb3xcw656E2MzPLylc8ZmaWla94zMwsKyceMzPLyonHzMyycuIx6+VUrMn2P0qPmyVdVYd2Jksas6vrNevIices9xsF/DnxRERLRHyyDu1Mplg/0KyunHjM6kwdVhiW9BZJS1NsqaQ3p3JzJV2lN65cfhnwNypW8/60ilXGb07HXKRilfT2Fco/WWr3DauAp/hLki5N/blH0qEq/mfUBym+NL1c0lvzniXrT5x4zOpI0ruA/w2cGBFHA58CvgXMj4h3Az8AysNmh1MsGPnfKBIOFKs6/DwixkbElRWaeSfFmmXjgAsl7ZFWa/gH4PiIGEuxPlf7kkf7Avek/twNfDQifkWxFuHnUju/20WnwOwNvGSOWX2dSIcVhiX9FfB3af+1FEudtOvOyuUd/SwiXgVelbQROJTOVwF/jWJ5IijWA/uvPXpmZj3kxGNWX9VW/S4r7+905fIqKq08XHEV8KS8wvkbVio2qzcPtZnV11LeuMLwr4Cz0v5/BH7RRR09WeG82irgu7ods5o58ZjVUUSsBtpXGH4I+AbF6sLnppWAP0zxuU9nVgBb0mSAT3ez3YeB9lXAVwBLKD4/6swC4HMq/nWyJxdY3XitNjMzy8pXPGZmlpUTj5mZZeXEY2ZmWTnxmJlZVk48ZmaWlROPmZll5cRjZmZZ/X+FA/UsoVu1qgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# continent\n",
    "g = sns.barplot(x = \"continent\", y = \"value\", data = comdata)\n",
    "comdata[[\"continent\", \"value\"]].groupby([\"continent\"]).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>position</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>DF</th>\n",
       "      <td>2.304348e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>GK</th>\n",
       "      <td>1.992073e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MF</th>\n",
       "      <td>3.121762e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST</th>\n",
       "      <td>3.330361e+06</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 value\n",
       "position              \n",
       "DF        2.304348e+06\n",
       "GK        1.992073e+06\n",
       "MF        3.121762e+06\n",
       "ST        3.330361e+06"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# position\n",
    "g = sns.barplot(x = \"position\", y = \"value\", data = comdata)\n",
    "comdata[[\"position\", \"value\"]].groupby([\"position\"]).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZ4AAAEHCAYAAACeFSCEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAW+klEQVR4nO3dfbBlVX3m8e9jI76hgtIoAZJmsMeIzthoD/SMOmPEgoaZTGONlFCp0OMw01FhIjGJYmpG4gs1akyoWKVMSOihcYxAiBZtqpX0IL4Wb82LQIvIFRFaQBoaEHTUgL/546weT1/P7ftin3U7fb+fqlNnn99ee611b7U87n3W3TtVhSRJvTxlvicgSVpYDB5JUlcGjySpK4NHktSVwSNJ6mqv+Z7A7m7//fevJUuWzPc0JOkfleuvv/7Bqlo8ap/BM40lS5awadOm+Z6GJP2jkuS7U+3zUpskqSuDR5LUlcEjSerK4JEkdWXwSJK6MngkSV0ZPJKkrgweSVJX/gGpunnnO9/J/fffzwtf+EI+/OEPz/d0JM0Tg0fd3H///Xzve9+b72lImmdeapMkdWXwSJK6MngkSV0ZPJKkrsYWPEmenuTaJF9PsjnJe1v90CTXJLkjycVJ9m71p7XPE23/kqG+3t3qtyc5dqi+stUmkpw5VJ/1GJKkPsZ5xvMT4HVV9XJgGbAyyQrgQ8A5VbUUeBg4tbU/FXi4ql4EnNPakeRw4CTgpcBK4ONJFiVZBHwMOA44HDi5tWW2Y0iS+hlb8NTA4+3jU9urgNcBl7b6OuCEtr2qfabtPzpJWv2iqvpJVX0HmACObK+Jqrqzqn4KXASsasfMdgxJUidj/Y6nnZncBDwAbAS+DTxSVU+0JluAg9r2QcA9AG3/o8Dzh+uTjpmq/vw5jDF53muSbEqyaevWrXP74SVJI401eKrqyapaBhzM4AzlJaOatfdRZx61C+s7G2PHQtV5VbW8qpYvXjzykeGSpDnqsqqtqh4BvgisAPZNsv2OCQcD97btLcAhAG3/c4Ftw/VJx0xVf3AOY0iSOhnnqrbFSfZt288AXg/cBlwJvLE1Ww1c1rbXt8+0/V+oqmr1k9qKtEOBpcC1wHXA0raCbW8GCxDWt2NmO4YkqZNx3qvtQGBdW332FOCSqvq7JN8ALkryAeBG4PzW/nzgE0kmGJyFnARQVZuTXAJ8A3gCOK2qngRIcjpwObAIWFtVm1tf75rNGJKkfsYWPFV1M3DEiPqdDL7vmVz/MXDiFH2dDZw9or4B2LArxpAk9eGdCyRJXflYBEkLns+K6svgkbTg+ayovgyeDl75hxfO9xR2C89+8DEWAXc/+Ji/E+D6PzllvqcgzQu/45EkdWXwSJK6MngkSV0ZPJKkrgweSVJXBo8kqSuDR5LUlX/HIy1gd7/vn833FHYLT2x7HrAXT2z7rr8T4Fffc8tY+/eMR5LUlcEjSerK4JEkdWXwSJK6MngkSV0ZPJKkrgweSVJXBo8kqSuDR5LUlXcukLTg7f/0nwFPtHeNm8Gjbn6297N2eJd2F3/wzx+Z7yksKAaPuvnh0mPmewqSdgNj+44nySFJrkxyW5LNSd7e6n+c5HtJbmqv44eOeXeSiSS3Jzl2qL6y1SaSnDlUPzTJNUnuSHJxkr1b/Wnt80Tbv2S6MSRJfYxzccETwO9X1UuAFcBpSQ5v+86pqmXttQGg7TsJeCmwEvh4kkVJFgEfA44DDgdOHurnQ62vpcDDwKmtfirwcFW9CDintZtyjPH9CiRJk40teKrqvqq6oW0/BtwGHLSTQ1YBF1XVT6rqO8AEcGR7TVTVnVX1U+AiYFWSAK8DLm3HrwNOGOprXdu+FDi6tZ9qDElSJ12WU7dLXUcA17TS6UluTrI2yX6tdhBwz9BhW1ptqvrzgUeq6olJ9R36avsfbe2n6mvyfNck2ZRk09atW2f980qSpjb24EmyD/C3wBlV9QPgXOAwYBlwH/Cn25uOOLzmUJ9LXzsWqs6rquVVtXzx4sUjDpEkzdVYgyfJUxmEzier6tMAVfX9qnqyqn4G/CU/v9S1BThk6PCDgXt3Un8Q2DfJXpPqO/TV9j8X2LaTviRJnYxzVVuA84HbqurPhuoHDjV7A3Br214PnNRWpB0KLAWuBa4DlrYVbHszWBywvqoKuBJ4Yzt+NXDZUF+r2/YbgS+09lONIUnqZJx/x/Mq4LeBW5Lc1Gp/xGBV2jIGl7juAn4HoKo2J7kE+AaDFXGnVdWTAElOBy4HFgFrq2pz6+9dwEVJPgDcyCDoaO+fSDLB4EznpOnGkCT1MbbgqaqvMvo7lQ07OeZs4OwR9Q2jjquqOxmxKq2qfgycOJsxJEl9eJNQSVJXBo8kqSuDR5LUlcEjSerK4JEkdWXwSJK6MngkSV0ZPJKkrgweSVJXBo8kqSuDR5LUlcEjSerK4JEkdWXwSJK6MngkSV0ZPJKkrgweSVJXBo8kqSuDR5LUlcEjSerK4JEkdWXwSJK6MngkSV0ZPJKkrsYWPEkOSXJlktuSbE7y9lZ/XpKNSe5o7/u1epJ8NMlEkpuTvGKor9Wt/R1JVg/VX5nklnbMR5NkrmNIkvoY5xnPE8DvV9VLgBXAaUkOB84ErqiqpcAV7TPAccDS9loDnAuDEAHOAo4CjgTO2h4krc2aoeNWtvqsxpAk9TO24Kmq+6rqhrb9GHAbcBCwCljXmq0DTmjbq4ALa+BqYN8kBwLHAhuraltVPQxsBFa2fc+pqquqqoALJ/U1mzEkSZ10+Y4nyRLgCOAa4AVVdR8Mwgk4oDU7CLhn6LAtrbaz+pYRdeYwhiSpk7EHT5J9gL8FzqiqH+ys6YhazaG+0+nM5Jgka5JsSrJp69at03QpSZqNsQZPkqcyCJ1PVtWnW/n72y9vtfcHWn0LcMjQ4QcD905TP3hEfS5j7KCqzquq5VW1fPHixTP/gSVJ0xrnqrYA5wO3VdWfDe1aD2xfmbYauGyofkpbebYCeLRdJrscOCbJfm1RwTHA5W3fY0lWtLFOmdTXbMaQJHWy1xj7fhXw28AtSW5qtT8CPghckuRU4G7gxLZvA3A8MAH8CHgzQFVtS/J+4LrW7n1Vta1tvxW4AHgG8Ln2YrZjSJL6GVvwVNVXGf2dCsDRI9oXcNoUfa0F1o6obwJeNqL+0GzHkCT14Z0LJEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSupo2eJK8IMn5ST7XPh/e7gggSdKszeSM5wIG90v7lfb5W8AZ45qQJGnPNpPg2b+qLgF+BlBVTwBPjnVWkqQ91kyC54dJnk97bs32uzqPdVaSpD3WTG4S+g4GjxM4LMnXgMXAG8c6K0nSHmva4KmqG5L8G+DFDO42fXtV/cPYZyZJ2iNNGzxJTplUekUSqurCMc1JkrQHm8mltn8xtP10Bs+5uQEweCRJszaTS23/dfhzkucCnxjbjCRJe7S53LngR8DSXT0RSdLCMJPveD5LW0rNIKgOBy4Z56QkSXuumXzH85Gh7SeA71bVljHNR5K0h5vJdzxf6jERSdLCMGXwJHmMn19i22EXUFX1nLHNSpK0x5oyeKrq2T0nIklaGGbyHQ8ASQ5g8Hc8AFTV3WOZkSRpjzaT5/H8+yR3AN8BvgTcBXxuBsetTfJAkluHan+c5HtJbmqv44f2vTvJRJLbkxw7VF/ZahNJzhyqH5rkmiR3JLk4yd6t/rT2eaLtXzLdGJKkfmbydzzvB1YA36qqQxncueBrMzjuAmDliPo5VbWsvTbA4OFywEnAS9sxH0+yKMki4GPAcQyWcZ/c2gJ8qPW1FHgY2P5wulOBh6vqRcA5rd2UY8zg55Ak7UIzCZ5/qKqHgKckeUpVXQksm+6gqvoysG2G81gFXFRVP6mq7wATwJHtNVFVd1bVT4GLgFVJArwOuLQdvw44YaivdW37UuDo1n6qMSRJHc0keB5Jsg/wFeCTSf6cwd/zzNXpSW5ul+L2a7WDgHuG2mxptanqzwceaQ+lG67v0Ffb/2hrP1VfkqSOZhI8Xwb2Bd4OfB74NvCbcxzvXOAwBmdM9wF/2uoZ0bbmUJ9LX78gyZokm5Js2rp166gmkqQ5mknwBLgc+CKwD3Bxu/Q2a1X1/ap6sqp+BvwlP7/UtQU4ZKjpwcC9O6k/COybZK9J9R36avufy+CS31R9jZrneVW1vKqWL168eC4/qiRpCtMGT1W9t6peCpwG/ArwpST/Zy6DJTlw6OMbgO0r3tYDJ7UVaYcyuAnptcB1wNK2gm1vBosD1ldVAVfy8yehrgYuG+prddt+I/CF1n6qMSRJHc3473iAB4D7gYeAA6ZrnORTwGuB/ZNsAc4CXptkGYNLXHcBvwNQVZuTXAJ8g8H3R6dV1ZOtn9MZnHEtAtZW1eY2xLuAi5J8ALgROL/Vzwc+kWSCwZnOSdONIUnqZyZ3p34r8CZgMYNVYv+lqr4x3XFVdfKI8vkjatvbnw2cPaK+Adgwon4nI1alVdWPgRNnM4YkqZ+ZnPH8GnBGVd007slIkvZ8M7k79ZnTtZEkaabm8gRSSZLmzOCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSV2MLniRrkzyQ5Nah2vOSbExyR3vfr9WT5KNJJpLcnOQVQ8esbu3vSLJ6qP7KJLe0Yz6aJHMdQ5LUzzjPeC4AVk6qnQlcUVVLgSvaZ4DjgKXttQY4FwYhApwFHAUcCZy1PUhamzVDx62cyxiSpL7GFjxV9WVg26TyKmBd214HnDBUv7AGrgb2TXIgcCywsaq2VdXDwEZgZdv3nKq6qqoKuHBSX7MZQ5LUUe/veF5QVfcBtPcDWv0g4J6hdltabWf1LSPqcxlDktTR7rK4ICNqNYf6XMb4xYbJmiSbkmzaunXrNN1Kkmajd/B8f/vlrfb+QKtvAQ4ZancwcO809YNH1Ocyxi+oqvOqanlVLV+8ePGsfkBJ0s71Dp71wPaVaauBy4bqp7SVZyuAR9tlssuBY5Ls1xYVHANc3vY9lmRFW812yqS+ZjOGJKmjvcbVcZJPAa8F9k+yhcHqtA8ClyQ5FbgbOLE13wAcD0wAPwLeDFBV25K8H7iutXtfVW1fsPBWBivnngF8rr2Y7RiSpL7GFjxVdfIUu44e0baA06boZy2wdkR9E/CyEfWHZjuGJKmf3WVxgSRpgTB4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSupqX4ElyV5JbktyUZFOrPS/JxiR3tPf9Wj1JPppkIsnNSV4x1M/q1v6OJKuH6q9s/U+0Y7OzMSRJ/cznGc9vVNWyqlrePp8JXFFVS4Er2meA44Cl7bUGOBcGIQKcBRwFHAmcNRQk57a2249bOc0YkqROdqdLbauAdW17HXDCUP3CGrga2DfJgcCxwMaq2lZVDwMbgZVt33Oq6qqqKuDCSX2NGkOS1Ml8BU8Bf5/k+iRrWu0FVXUfQHs/oNUPAu4ZOnZLq+2svmVEfWdj7CDJmiSbkmzaunXrHH9ESdIoe83TuK+qqnuTHABsTPLNnbTNiFrNoT5jVXUecB7A8uXLZ3WsJGnn5uWMp6rube8PAJ9h8B3N99tlMtr7A635FuCQocMPBu6dpn7wiDo7GUOS1En34EnyrCTP3r4NHAPcCqwHtq9MWw1c1rbXA6e01W0rgEfbZbLLgWOS7NcWFRwDXN72PZZkRVvNdsqkvkaNIUnqZD4utb0A+Exb4bwX8NdV9fkk1wGXJDkVuBs4sbXfABwPTAA/At4MUFXbkrwfuK61e19VbWvbbwUuAJ4BfK69AD44xRiSpE66B09V3Qm8fET9IeDoEfUCTpuir7XA2hH1TcDLZjqGJKmf3Wk5tSRpATB4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHVl8EiSujJ4JEldGTySpK4MHklSVwaPJKkrg0eS1JXBI0nqyuCRJHW1IIMnycoktyeZSHLmfM9HkhaSBRc8SRYBHwOOAw4HTk5y+PzOSpIWjgUXPMCRwERV3VlVPwUuAlbN85wkacHYa74nMA8OAu4Z+rwFOGq4QZI1wJr28fEkt3ea20KwP/DgfE9id5CPrJ7vKWhH/tvc7qzsil5+baodCzF4Rv1Ga4cPVecB5/WZzsKSZFNVLZ/veUiT+W+zn4V4qW0LcMjQ54OBe+dpLpK04CzE4LkOWJrk0CR7AycB6+d5TpK0YCy4S21V9USS04HLgUXA2qraPM/TWki8hKndlf82O0lVTd9KkqRdZCFeapMkzSODR5LUlcGjXS7J4zNo87tJbkvyySSvTfKvesxNC1uSDUn2nabNF5P8wrLqJMuSHD++2S0cBo/my9uA46vqt4DXAgaPxipJgH9XVY/MsYtlgMGzCxg8Gqskf5jkuiQ3J3lvq/1P4J8A65P8HvAW4PeS3JTkNfM5X+1ZkixpZ9YfB24Ankyyf9v335N8M8nGJJ9K8gdDh56Y5Nok30rymvanF+8D3tT+nb5pHn6cPcaCW06tfpIcAyxlcH+8MAiaf11Vb0myEviNqnowyXOBx6vqI/M5X+2xXgy8uareluQugHYp7T8ARzD47+ANwPVDx+xVVUe2S2tnVdXrk7wHWF5Vp/ed/p7H4NE4HdNeN7bP+zAIoi/P24y0EH23qq6eVHs1cFlV/V+AJJ+dtP/T7f16YMl4p7fwGDwapwD/o6r+Yr4nogXthyNq090F8yft/Un87+Qu53c8GqfLgf+UZB+AJAclOWBEu8eAZ3edmRa6rwK/meTp7d/nv53BMf473UUMHo1NVf098NfAVUluAS5l9P9wPwu8wcUF6qWqrmNwj8avM7istgl4dJrDrgQOd3HBL89b5khakJLsU1WPJ3kmg+8d11TVDfM9r4XAa5eSFqrz2mPvnw6sM3T68YxHktSV3/FIkroyeCRJXRk8kqSuDB5JUlcGj7SbSbI4yTVJbvxl/64pya+3vzu5Mclhszx23yRv+2XGl0YxeKR5kGTRTnYfDXyzqo6oqq/8kv2dwOCeZEdU1bdnOc19GTy+QtqlDB5pF2u34v9mknXtcRCXJnlmkruSvCfJVxncdv+wJJ9Pcn2Sr7Szk2XAh4Hj25nKM5Ick+SqJDck+ZuhWxDt0N+IeRwPnAH85yRXtto7ktzaXmcMtR1V/yBwWJvHn4z3t6aFxD8glcbjxcCpVfW1JGv5+ZnDj6vq1QBJrgDeUlV3JDkK+HhVvW749vvt2TH/DXh9Vf0wybuAdzB4NswO/U1WVRvas48er6qPJHkl8GbgKAY3ybwmyZcY/B/QUfUzgZdV1bJd/LvRAmfwSONxT1V9rW3/b+B32/bFMLhdC4Onrv7N4MGYADxtRD8rgMOBr7V2ewNXDe2/eBZzejXwmar6YZvDp4HXMAibUfX1s+hbmjGDRxqPybcE2f55+y36nwI8MoOziQAbq+rkKfaPuuX/zvqaTV0aC7/jkcbjV5P8y7Z9MoPb8P9/VfUD4DtJTgTIwMtH9HM18KokL2rtnpnkn85xTl8GTmh9PAt4A/CVndR9DIDGwuCRxuM2YHWSm4HnAeeOaPNbwKlJvg5sBlZNblBVW4H/CHyq9XU18OtzmVC7CeYFwLXANcBfVdWNO6k/xOAS360uLtCu5E1CpV0syRLg76rqZfM8FWm35BmPJKkrz3ikPUCSjwGvmlT+86r6X/MxH2lnDB5JUldeapMkdWXwSJK6MngkSV0ZPJKkrv4fYPu4JD1Dk1YAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# prefer_foot\n",
    "g = sns.barplot(x = \"prefer_foot\", y = \"value\", data = comdata)\n",
    "comdata[[\"prefer_foot\", \"value\"]].groupby([\"prefer_foot\"]).mean()\n",
    "\n",
    "# ttest로 평균차이 확인\n",
    "prefter_left = comdata.loc[comdata[\"prefer_foot\"] == \"left\", \"value\"]\n",
    "prefter_left = prefter_left.dropna()\n",
    "\n",
    "prefter_right = comdata.loc[comdata[\"prefer_foot\"] == \"right\", \"value\"]\n",
    "prefter_right = prefter_right.dropna()\n",
    "\n",
    "stats.ttest_ind(prefter_left, prefter_right)\n",
    "# pvalue > 0.05, 왼발과 오른발에는 평균에는 차이가 없다\n",
    "# 그럼므로 주사용 발에 따른 선수가치에는 차이가 없으므로 선수가치를 판단할때는 제거한다.\n",
    "comdata = comdata.drop([\"prefer_foot\"], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   value\n",
      "reputation              \n",
      "1.0         1.690092e+06\n",
      "2.0         8.639221e+06\n",
      "3.0         2.201483e+07\n",
      "4.0         3.342903e+07\n",
      "5.0         6.062500e+07\n",
      "[5. 4. 3. 1. 2.]\n",
      "count    12760.000000\n",
      "mean         1.134796\n",
      "std          0.431366\n",
      "min          1.000000\n",
      "25%          1.000000\n",
      "50%          1.000000\n",
      "75%          1.000000\n",
      "max          5.000000\n",
      "Name: reputation, dtype: float64\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAERCAYAAAB2CKBkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAATVUlEQVR4nO3df7DldX3f8eeLxZUKCBN3zVoWhTEobi2C3iLNtkoU25U27ExLGkjUmKHupCNEI80OaR1U8k+zNrW1YlMaLcEmkDVpko1dJa3yw2FAuYBigGI2iHIXb3dRQaMJsvDuH+e77PHsubsX9n7PvXs/z8fMzvn++Nxz3vOZPfd1v5/v9/v5pqqQJLXriMUuQJK0uAwCSWqcQSBJjTMIJKlxBoEkNc4gkKTGHZZBkOTjSXYl+fN5tH1xkhuS3JXk7iTnTqJGSTpcHJZBAFwNbJhn2/cCW6vqDOAC4KN9FSVJh6PDMgiq6mbg28Pbkrw0yWeS3JHk80lO3dsceH63fBzw8ARLlaQl78jFLmABXQX8UlX9RZLXMvjL/w3A+4E/S3IJcDRwzuKVKElLz7IIgiTHAD8JfDLJ3s3P7V4vBK6uqt9M8veBTyR5ZVU9tQilStKSsyyCgMEQ16NVdfqYfRfRnU+oqluTHAWsAnZNsD5JWrIOy3MEo6rqu8DXkvwMQAZe1e3+BvDGbvsrgKOA3YtSqCQtQb0FwcEu8ex+WX84yY7uss5XP4P3vha4FXh5kpkkFwE/D1yU5MvAPcDGrvmlwDu67dcCby+nXJWkp6Wv34lJXgf8FXBNVb1yzP5zgUuAc4HXAv+pql7bSzGSpDn1dkQw7hLPERsZhERV1W3A8Ule1Fc9kqTxFvNk8QnAQ0PrM922b442TLIJ2ARw9NFHv+bUU08dbSJJOoA77rjjkapaPW7fYgZBxmwbO05VVVcxuE+Aqampmp6e7rMuSVp2knx9rn2LedXQDHDi0PpavOtXkiZuMYNgG/C27uqhs4DHqmq/YSFJUr96GxrqLvE8G1iVZAZ4H/AcgKr6LWA7gyuGdgA/AH6xr1okSXPrLQiq6sKD7C/gnX19viRpfpbFncWSpGfPIJCkxhkEktQ4g0CSGrdcpqGWpCVn8+bNzM7OsmbNGrZs2bLY5czJIJCknszOzrJz587FLuOgHBqSpMYZBJLUOINAkhpnEEhS4wwCSWqcQSBJjTMIJKlxBoEkNc4gkKTGGQSS1DiDQJIaZxBIUuMMAklqnEEgSY0zCCSpcQaBJDXOIJCkxhkEktQ4g0CSGmcQSFLjDAJJapxBIEmNMwgkqXEGgSQ1ziCQpMYZBJLUOINAkhpnEEhS4wwCSWpcr0GQZEOS+5PsSHLZmP0vTnJDkruS3J3k3D7rkSTtr7cgSLICuBJ4M7AOuDDJupFm7wW2VtUZwAXAR/uqR5I0Xp9HBGcCO6rqgar6IXAdsHGkTQHP75aPAx7usR5J0hh9BsEJwEND6zPdtmHvB96SZAbYDlwy7o2SbEoynWR69+7dfdQqSc3qMwgyZluNrF8IXF1Va4FzgU8k2a+mqrqqqqaqamr16tU9lCpJ7eozCGaAE4fW17L/0M9FwFaAqroVOApY1WNNkqQRfQbB7cApSU5OspLByeBtI22+AbwRIMkrGASBYz+SNEG9BUFV7QEuBq4H7mNwddA9Sa5Icl7X7FLgHUm+DFwLvL2qRoePJEk9OrLPN6+q7QxOAg9vu3xo+V5gfZ81SJIOzDuLJalxBoEkNc4gkKTGGQSS1DiDQJIaZxBIUuMMAklqnEEgSY0zCCSpcQaBJDXOIJCkxhkEktQ4g0CSGmcQSFLjDAJJapxBIEmNMwgkqXEGgSQ1ziCQpMYZBJLUOINAkhpnEEhS4wwCSWqcQSBJjTMIJKlxBoEkNc4gkKTGGQSS1DiDQJIaZxBIUuMMAklqnEEgSY0zCCSpcQaBJDWu1yBIsiHJ/Ul2JLlsjjb/Ism9Se5J8nt91iNJ2t+Rfb1xkhXAlcCbgBng9iTbqureoTanAL8GrK+q7yR5YV/1SJLG6y0IgDOBHVX1AECS64CNwL1Dbd4BXFlV3wGoql091iNJT/vIpX/a+2c8+sj3n36dxOdd/Js//ax+rs+hoROAh4bWZ7ptw14GvCzJLUluS7Jh3Bsl2ZRkOsn07t27eypXktrUZxBkzLYaWT8SOAU4G7gQ+O0kx+/3Q1VXVdVUVU2tXr16wQuVpJb1GQQzwIlD62uBh8e0+ZOqeqKqvgbczyAYJEkT0mcQ3A6ckuTkJCuBC4BtI23+GPgpgCSrGAwVPdBjTZKkEb0FQVXtAS4GrgfuA7ZW1T1JrkhyXtfseuBbSe4FbgB+taq+1VdNkqT99XnVEFW1Hdg+su3yoeUC3tP9kyQtAu8slqTGGQSS1DiDQJIaZxBIUuMMAklqnEEgSY07aBAk+fEkH0vy6W59XZKL+i9NkjQJ8zkiuJrBjV9/u1v/KvDuvgqSJE3WfIJgVVVtBZ6Cp+8YfrLXqiRJEzOfIPh+khfQzRya5CzgsV6rkiRNzHymmHgPg8niXprkFmA1cH6vVUmSJuagQVBVdyZ5PfByBs8YuL+qnui9MknSRBw0CJK8bWTTq5NQVdf0VJMkaYLmMzT094aWjwLeCNwJGASStAzMZ2jokuH1JMcBn+itIknSRD2bO4t/gI+TlKRlYz7nCP6UfQ+dPwJYB2ztsyhJ0uTM5xzBvx9a3gN8vapmeqpHkjRh8zlHcNMkCpEkLY45gyDJ99g3JPQjuxg8bvj5vVUlSZqYOYOgqo6dZCGSpMUxn3MEACR5IYP7CACoqm/0UpEkaaLm8zyC85L8BfA14CbgQeDTPdclSZqQ+dxH8OvAWcBXq+pkBncW39JrVZKkiZlPEDxRVd8CjkhyRFXdAJzec12SpAmZzzmCR5McA3we+N0kuxjcTyBJWgbmc0RwM3A88C7gM8BfAj/dZ1GSpMmZTxCEwTOLbwSOAX6/GyqSJC0DBw2CqvpAVf0d4J0MHmB/U5L/03tlkqSJeCazj+4CZoFvAS/spxxJ0qTN5z6Cf5XkRuCzwCrgHVV1Wt+FSZImYz5XDb0EeHdVfanvYiRJkzef2Ucvm0QhkqTF8WyeUCZJWkYMAklqXK9BkGRDkvuT7Egy5xBTkvOTVJKpPuuRJO1v3tNQP1NJVgBXAm8CZoDbk2yrqntH2h0L/DLwhb5qkTQ5mzdvZnZ2ljVr1rBly5bFLkfz0OcRwZnAjqp6oKp+CFwHbBzT7teBLcDf9FiLpAmZnZ1l586dzM7OLnYpmqc+g+AE4KGh9Zlu29OSnAGcWFWfOtAbJdmUZDrJ9O7duxe+UklqWJ9BkDHbnn4GcpIjgA8Blx7sjarqqqqaqqqp1atXL2CJkqQ+g2AGOHFofS3w8ND6scArgRuTPMjg4TfbPGEsSZPVZxDcDpyS5OQkK4ELgG17d1bVY1W1qqpOqqqTgNuA86pquseaJEkjeguCqtoDXMxgCuv7gK1VdU+SK5Kc19fnSpKemd4uHwWoqu3A9pFtl8/R9uw+a5EkjeedxZLUOINAkhpnEEhS4wwCSWqcQSBJjTMIJKlxBoEkNc4gkKTG9XpDmaSl5abXvb73z/jrI1dAwl/PzEzk815/8029f8Zy5xGBJDXOIJCkxhkEktQ4g0CSGmcQSFLjDAJJapxBIEmNMwgkqXEGgSQ1ziCQpMYZBJLUOINAkhpnEEhS4wwCSWqcQSBJjfN5BJIW1PFVP/Kqpc8gkLSg3vLkU4tdwpJx9Mrn/8jrUmUQSFJP1r/0ny12CfNiEEgLYPPmzczOzrJmzRq2bNmy2OVIz4hBIC2A2dlZdu7cudhlSM+KVw1JUuMMAklqnEEgSY0zCCSpcZ4s1rK3/j+v7/0zVj66kiM4gocefWgin3fLJbf0/hlqR69HBEk2JLk/yY4kl43Z/54k9ya5O8lnk7ykz3okSfvrLQiSrACuBN4MrAMuTLJupNldwFRVnQb8AeAF2JI0YX0eEZwJ7KiqB6rqh8B1wMbhBlV1Q1X9oFu9DVjbYz2SpDH6DIITgIeG1me6bXO5CPj0uB1JNiWZTjK9e/fuBSxRktRnEGTMtrHTESZ5CzAFfHDc/qq6qqqmqmpq9erVC1iiJKnPq4ZmgBOH1tcCD482SnIO8G+B11fV4z3WI/Wmnlc8xVPU85x6WYefPoPgduCUJCcDO4ELgJ8bbpDkDOC/AhuqalePtUi9emL9E4tdgvSs9TY0VFV7gIuB64H7gK1VdU+SK5Kc1zX7IHAM8MkkX0qyra96JEnj9XpDWVVtB7aPbLt8aPmcPj9fknRwTjEhSY0zCCSpcQaBJDXOIJCkxhkEktQ4g0CSGmcQSFLjDAJJapxBIEmNMwgkqXEGgSQ1ziCQpMYZBJLUOINAkhpnEEhS4wwCSWqcQSBJjev1CWVa/jZv3szs7Cxr1qxhy5Yti12OpGfBINAhmZ2dZefOnYtdhqRDYBAsU9+44u9O5HP2fPvHgCPZ8+2v9/6ZL778K72+v9QqzxFIUuM8ItAhWXXUU8Ce7lXS4cgg0CH516c9utglSDpEDg1JUuMMAklqnEEgSY0zCCSpcQaBJDXOIJCkxhkEktQ4g0CSGmcQSFLjDAJJapxBIEmNW1ZzDb3mV69Z7BIW3B0ffNtilyBpmev1iCDJhiT3J9mR5LIx+5+b5Pe7/V9IclKf9UiS9tdbECRZAVwJvBlYB1yYZN1Is4uA71TVTwAfAn6jr3okSeP1eURwJrCjqh6oqh8C1wEbR9psBH6nW/4D4I1J0mNNkqQRqap+3jg5H9hQVf+yW38r8NqquniozZ93bWa69b/s2jwy8l6bgE3d6suB+3sp+plZBTxy0FZtsC8G7Id97It9lkpfvKSqVo/b0efJ4nF/2Y+mznzaUFVXAVctRFELJcl0VU0tdh1LgX0xYD/sY1/sczj0RZ9DQzPAiUPra4GH52qT5EjgOODbPdYkSRrRZxDcDpyS5OQkK4ELgG0jbbYBv9Atnw98rvoaq5IkjdXb0FBV7UlyMXA9sAL4eFXdk+QKYLqqtgEfAz6RZAeDI4EL+qqnB0tqqGqR2RcD9sM+9sU+S74vejtZLEk6PDjFhCQ1ziCQpMYZBAeQ5ONJdnX3O4zbnyQf7qbIuDvJqydd4yQkOTHJDUnuS3JPkneNadNKXxyV5ItJvtz1xQfGtGlq6pQkK5LcleRTY/Y10xdJHkzylSRfSjI9Zv+S/Y4YBAd2NbDhAPvfDJzS/dsE/JcJ1LQY9gCXVtUrgLOAd46ZLqSVvngceENVvQo4HdiQ5KyRNq1NnfIu4L459rXWFz9VVafPcd/Akv2OGAQHUFU3c+D7GjYC19TAbcDxSV40meomp6q+WVV3dsvfY/ClP2GkWSt9UVX1V93qc7p/o1dcNDN1SpK1wD8BfnuOJs30xTws2e+IQXBoTgAeGlqfYf9fkMtKd2h/BvCFkV3N9EU3FPIlYBfwv6tqzr6oqj3AY8ALJlvlxPxHYDPw1Bz7W+qLAv4syR3dtDijlux3xCA4NPOaImO5SHIM8IfAu6vqu6O7x/zIsuyLqnqyqk5ncLf8mUleOdKkib5I8k+BXVV1x4Gajdm27Pqis76qXs1gCOidSV43sn/J9oVBcGjmM43GspDkOQxC4Her6n+OadJMX+xVVY8CN7L/eaRWpk5ZD5yX5EEGswu/Icn/GGnTSl9QVQ93r7uAP2IwA/OwJfsdMQgOzTbgbd3VAGcBj1XVNxe7qIXWjel+DLivqv7DHM1a6YvVSY7vlv8WcA7wf0eaNTF1SlX9WlWtraqTGMwK8LmqestIsyb6IsnRSY7duwz8I2D0asMl+x1ZVo+qXGhJrgXOBlYlmQHex+DkIFX1W8B24FxgB/AD4BcXp9LerQfeCnylGxsH+DfAi6G5vngR8Dvdg5eOALZW1aeW0dQph6zRvvhx4I+68+BHAr9XVZ9J8kuw9L8jTjEhSY1zaEiSGmcQSFLjDAJJapxBIEmNMwgkqXEGgXQIkpyU5OeeabskU0k+3G910vwYBGpOd0PPQv3fPwk4aBCMtquq6ar65QWqQTokBoGa0P1Ffl+SjwJ3Am9NcmuSO5N8sptHae+c8r/RPXPgi0l+ott+dZLzh95v7wyk/w74h90c9L/Sfc7nu/e9M8lPztHu7L3z9yf5sSR/3M1Rf1uS07rt78/gmRg3JnkgicGhXhgEasnLgWuANzGYJ/+cbpKwaeA9Q+2+W1VnAh9hMLvmgVwGfL6bg/5DDGYkfVP3vj8LfHiOdsM+ANxVVacxuGP7mqF9pwL/mMG8Ne/r5nySFpRTTKglX6+q27pZM9cBt3RTAqwEbh1qd+3Q6+gv7YN5DvCRJKcDTwIvm8fP/APgnwNU1eeSvCDJcd2+/1VVjwOPJ9nFYCqDmWdYk3RABoFa8v3uNQyeI3DhHO1qzPIeuiPobhK+lXP87K8A/w94Vdf+b+ZR14GmJ358aNuT+J1VDxwaUotuA9YPjf8/L8nwX+4/O/S690jhQeA13fJGuskHge8Bxw797HHAN6vqKQYT9a2Yo92wm4Gf72o5G3hkzPMepN7414WaU1W7k7wduDbJc7vN7wW+2i0/N8kXGPyhtPeo4b8Bf5Lki8Bn2Xd0cTewJ8mXGTzj+qPAHyb5GeCGA7S7a6ik9wP/PcndDGal/AWkCXL2UWlI95CVqap6ZLFrkSbFoSFJapxHBJLUOI8IJKlxBoEkNc4gkKTGGQSS1DiDQJIa9/8BMJBPGAErlDYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# reputation\n",
    "g = sns.barplot(x = \"reputation\", y = \"value\", data = comdata)\n",
    "print(comdata[[\"reputation\", \"value\"]].groupby([\"reputation\"]).mean())\n",
    "print(comdata[\"reputation\"].unique())\n",
    "print(comdata[\"reputation\"].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                         value\n",
      "stat_skill_moves              \n",
      "1.0               1.992073e+06\n",
      "2.0               1.448798e+06\n",
      "3.0               3.128878e+06\n",
      "4.0               1.100068e+07\n",
      "5.0               2.005161e+07\n",
      "[4. 1. 3. 2. 5.]\n",
      "count    12760.000000\n",
      "mean         2.405643\n",
      "std          0.777576\n",
      "min          1.000000\n",
      "25%          2.000000\n",
      "50%          2.000000\n",
      "75%          3.000000\n",
      "max          5.000000\n",
      "Name: stat_skill_moves, dtype: float64\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# stat_skill_moves\n",
    "g = sns.barplot(x = \"stat_skill_moves\", y = \"value\", data = comdata)\n",
    "print(comdata[[\"stat_skill_moves\", \"value\"]].groupby([\"stat_skill_moves\"]).mean())\n",
    "print(comdata[\"stat_skill_moves\"].unique())\n",
    "print(comdata[\"stat_skill_moves\"].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 576x432 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAFgCAYAAACFYaNMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3de3TdZ33n+/d3X3SXZdmW4mArsRwSTALOBediSEMAz0ygPcnMrAyNGdrTc6BJu6CBTjszsOgE6q6ZQuFMh8xAx26Gw5ShXJoznXpoAjSE4KbYSZwruTiOIzuxnNiSJVnWfd++54/flrz31tbFkrZ+W3t/Xms50v7pp62vA/7k8fN7nu9j7o6IiCy/SNgFiIhUKwWwiEhIFMAiIiFRAIuIhEQBLCISEgWwiEhIVmQAm9k3zKzHzJ6fx70XmdlPzexpM3vOzD60HDWKiMxlRQYw8E3glnne+wfA9939auAO4OulKkpE5HysyAB2931Af+41M7vEzH5oZk+a2d+b2ZbJ24FV2c9bgDeWsVQRkRnFwi5gCe0BfsvdXzGz6wlGuu8HvgD82Mx+B2gEdoRXoojIORURwGbWBLwb+Cszm7xcm/24E/imu/8/ZrYd+JaZvcPdMyGUKiIypSICmGAq5Yy7X1Xkax8jO1/s7vvNrA5YB/QsY30iItOsyDngQu5+FjhqZv8CwAJXZr/8OvCB7PW3A3VAbyiFiojksJXYDc3MvgPcTDCSPQV8HngY+DPgQiAOfNfdd5nZ5cCfA00ED+T+jbv/OIy6RURyrcgAFhGpBBUxBSEishKtuIdwt9xyi//whz8MuwwRkfNhxS6uuBHw6dOnwy5BRGRJrLgAFhGpFApgEZGQKIBFREKiABYRCYkCWEQkJApgEZGQKIBFREKiABYRCYkCWEQkJCtuK7KIyPl45FAPu/d1cXxglI7WBu66aTM3b2kPuyxAI2ARqWCPHOrhnr0v0DM0zur6OD1D49yz9wUeOVQe5zEogEWkYu3e10U8ajTUxDALPsajxu59XWGXBiiARaSCHR8YpT4ezbtWH4/SPTAaUkX5FMAiUrE6WhsYS6bzro0l02xsbQiponwKYBGpWHfdtJlk2hlNpHAPPibTzl03bQ67NEABLCIV7OYt7ey69Qram+sYHEvS3lzHrluvKJtVEFqGJiIV7eYt7WUTuIU0AhYRCYkCWEQkJApgEZGQKIBFREKiABYRCYkCWEQkJApgEZGQKIBFREKiABYRCYkCWEQkJApgEZGQKIBFREKiABYRCYkCWEQkJApgEZGQlCyAzewbZtZjZs/P8HUzs3vN7IiZPWdm15SqFhGRclTKEfA3gVtm+foHgUuzv+4E/qyEtYiIlJ2SBbC77wP6Z7nlNuAvPHAAWG1mF5aqHhGRchPmHPAG4HjO6+7stWnM7E4zO2hmB3t7e5elOBGRUgszgK3INS92o7vvcfdt7r6tra2txGWJiCyPMAO4G+jIeb0ReCOkWkREll2YAbwX+PXsaogbgEF3fzPEekREllXJjqU3s+8ANwPrzKwb+DwQB3D3/wo8AHwIOAKMAv9XqWoRESlHJQtgd985x9cd+ESpfr6ISLnTTjgRkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQkCmARkZAogEVEQqIAFhEJiQJYRCQksVK+uZndAnwViAL3ufsXC75+EfDfgdXZez7j7g+UsiYRWXkeOdTD7n1dHB8YpaO1gbtu2szNW9rDLmvRSjYCNrMo8DXgg8DlwE4zu7zgtj8Avu/uVwN3AF8vVT0isjI9cqiHe/a+QM/QOKvr4/QMjXPP3hd45FBP2KUtWimnIK4Djrh7l7sngO8CtxXc48Cq7OctwBslrEdEVqDd+7qIR42Gmhhmwcd41Ni9ryvs0hatlAG8ATie87o7ey3XF4CPmlk38ADwO8XeyMzuNLODZnawt7e3FLWKSJk6PjBKfTyad60+HqV7YDSkipZOKQPYilzzgtc7gW+6+0bgQ8C3zGxaTe6+x923ufu2tra2EpQqIuWqo7WBsWQ679pYMs3G1oaQKlo6pQzgbqAj5/VGpk8xfAz4PoC77wfqgHUlrElEVpi7btpMMu2MJlK4Bx+TaeeumzaHXdqilTKAnwAuNbNOM6sheMi2t+Ce14EPAJjZ2wkCWHMMIjLl5i3t7Lr1Ctqb6xgcS9LeXMeuW6+oiFUQJVuG5u4pM/sk8COCJWbfcPcXzGwXcNDd9wK/B/y5mf0uwfTEb7h74TSFiFS5m7e0V0TgFrKVlnfbtm3zgwcPhl2GiMj5KPZMTDvhRETCogAWEQmJAlhEJCQKYBGRkCiARURCogAWEQlJSdtRiohMqtSWkouhEbCIlFwlt5RcDAWwiJRcJbeUXAxNQYhIyR0fGGV1fTzv2vm0lKzU6QuNgEWk5BbTUrKSpy8UwCJScotpKVnJ0xcKYBEpucW0lKzkEzE0Bywiy2KhLSU7WhvoGRqnoeZcXOlEDBGRZVDJJ2JoBCwiy+Lehw5z36NHGUmkaayJ8vEbO7l7x2Vzft/NW9rZRTAX3D0wysYKWgWhABaRkrv3ocN89eEjRAxikWAK4asPHwGYdwhXQuAW0hSEiJTcfY8ezYZvhIhFsh+D69VMI2ARKbmRRJpYwXAvYsH1QpW66aIYjYBFpOQaa6JkCo6fzHhwPVclb7ooRgEsIiX38Rs7yTikMhkynsl+DK7nquRNF8VoCkJESm7yQdtcqyAW2zNipVEAi8iyuHvHZXOueKjkTRfFaApCRMpGJW+6KEYBLCJlYzE9I1YiTUGISFmp1E0XxWgELCISEo2ARWTJVdNmisXQCFhEllS1baZYDI2ARWRJ5W6mAGioiTGaSLF7X9e0UXC1j5Q1AhaRJTXfEyw0UtYIWESWWEdrA0dPDzM0niKRzlATjdBcF6NzXVPefeczUq5UCmARWVLbN6/h8WP9RCzoeJZIZ+gdTrB9cw079xyYmm44fOosF7bU531vJW87LkZTECKypPZ39dPeXENNNELGoSYaoaUuxgPPn8qbbhieSHN6eCLveyt523ExGgGLyJI6PjDK2sZa1jXVTV17tWeIdMbzphvWNMbpH0nSWBujPh5lLJmu6G3HxWgELCJLqqO1gbFkfqP1iXSG2oKO7Gsba2mui1XNtuNiShrAZnaLmb1sZkfM7DMz3PNhM3vRzF4ws78sZT0iUnrFGurEIsGDuFxjyTRtTbUA5PZqf+RQDzv3HODGLz3Mzj0HKnpVhLn73Hct5I3NosBh4B8B3cATwE53fzHnnkuB7wPvd/cBM2t391n/bW/bts0PHjxYkppFZGlMru+dPMV4++Y1fOvAawyNp0hlMsQiEWrjEeLRCC318akpiLNjSRzyriXTXgkjYyt2sZRzwNcBR9y9C8DMvgvcBryYc89vAl9z9wGAucJXRFaGwoY6jxzqYTzZRSIdnISR8QyJVIb2VbV588InBsbAmFodUelL00o5BbEBOJ7zujt7LddlwGVm9g9mdsDMbin2RmZ2p5kdNLODvb29JSpXRErliw++xFgyQzwSoS4WIR6JkAEGRhJ596UyGdIFh8dV8tK0UgZwsSF34XxHDLgUuBnYCdxnZqunfZP7Hnff5u7b2tralrxQESmto32jwbrgiGFmwUcgkc6PhFgkQjSSHx2VvDStlFMQ3UBHzuuNwBtF7jng7kngqJm9TBDIT5SwLhEpA9EIpDLwyqmhafPCo4lUVSxNK+UI+AngUjPrNLMa4A5gb8E9/wt4H4CZrSOYkqjM409FqtjmdY3ZuV/H8eCjQ9QAAzMDg3g0wq/fcHHVLE0r2QjY3VNm9kngR0AU+Ia7v2Bmu4CD7r43+7V/bGYvAmngX7t7X6lqEpHlUdjl7IPvWM+Jv+9iOJEm48EWZYD25lrams9t2BhNpNjf1c937rwhpMqXV8mWoZWKlqGJlLfJLmfxqOUtL5tIpkmkfWq6IZHO0NFaz6r6mqnvdXcGx5L8/b99f4i/g5JY9mVoIlKFinU5m1xedukFzVP3vXJqiFNDE3kBXMkP3IrRVmQRWVLF+gEXW152waraqjqCvhiNgEVkwYqdaNHR2kDP0PjUCBiC5WWFfwmPRSNc1t7E6oaaqR1z1XYihgJYRBYkd64390SL26/ZwP1PnchbStZcF8Nh2vKyf/fLW6oqcAtpCkJEFiR3rtcs+BiPGvu7+tl16xV5S8m+fPuVfOX2K6tmedl8zTkCNrMLgP8AvMXdP2hmlwPb3f2/lbw6ESkbhdMNs51oUdgLYlK1B26h+UxBfBP4f4HPZV8fBr4HKIBFqkSx6YbJEy1y1/Ge7yoGnYo8t3Xu/n0gA8EGC4JNEyJSJYpNN6xpjDMwmlzwKgadijy/AB4xs7VkG+mY2Q3AYEmrEpGyUmxp2WJPtJhpDnn3vurpRjCfKYh/RdDD4RIz+wegDbi9pFWJSFkptrRsphMt5uv4wCir6+N51yq59WQxcwawuz9lZu8F3kawku/lbPcyEakSd920mXv2vpC3jGzy9IpEOpM3hbCL+T1s62ht4OjpYYbGUyTSGWqiwbFFneuaSv77KRfzWQXx6wWXrjEz3P0vSlSTiJSZm7e0swvyjhmKR4xkwUnHk6dXkL13todr2zev4fFj/UGfYAuCvHc4wUeuWzPt51fqw7o5m/GY2X/OeVkHfAB4yt1DmYZQMx6R8nDjlx5mdX08aCWZ5e6cPDs+NZ8727luO/cc4FjfMGfHzo2AV9XH2LS2Ka8bWrHmPivwnLiFNeNx99/JexezFuBbS1SUiKwQ9z50mPsePcpIIk1jTZRVtUEYFs4LJ1IZWuqt6Mg4NzCPD4yytrGWdU3nlrG5+7Q54GLNfSrlnLiF7IQbJTi1QkSqxL0PHearDx9hLJkmFgmC9o2zE/ScHZ+2DG1ypJqr2MO1jtYGxpL5K1qLrSMutgKjUh7WzRnAZva/zWxv9tcPgJeBvyl9aSJSLu579CgRC5rqRCwydXZb2pm2DO2yC1bNK1jvumnzvLqhzTeoV6L5LEP7Ss7nKeA1d+8uUT0iUoZGEsHIN1fEYCKVKXp6ReGKiWLBWuzBXrGHa8VWYFRK28r5zAH/bDkKEZHy1VgTBF/ugcUZD64Xmm+wTt471zzu+bzfSjNjAJvZEMXXVxvg7r6qZFWJSFn5+I2dfPXhI6QyGSJG9oDN4Hox8wnW87HU71cuZgxgd2+e6WsiUl3u3nEZQN4qiI/f2Dl1XRZm3odymlk7wTpgANz99VIVNRutAxaRFajoOuD5rIK41cxeAY4CPwOOAQ8uaWkiIlVoPqsg/gi4AXjI3a82s/cBO0tbloiUm2LbgaH4luPCTRuarihuPluRD7r7NjN7Frja3TNm9ri7X7c8JebTFITI8iu2HXiyGU9LfTxvedi7Lmph73Mnp3o8TD6wu3Xrek6eTVRcP4d5WthWZOCMmTUBfw9828x6CNYDi0iVKLYd+MTAGBl3xhLpvG5mk+EbiwQznEGjnTT/69k32byucUGd0yrVfAJ4H7Aa+BTwUaAF2FXKokSkvBwfGGUimeLo6REyfm5kC5BMp3EglU4znkqTzkAkAhOpNO5gOfdWYj+HxZhPLwgDfgQ8AjQB33P3vlIWJSJlJpOhdzg5FaSZnJlLz/mYzgSfJzPgDlj2I0EQ56qUfg6LMZ+dcH8I/KGZbQV+FfiZmXW7+46SVyciZaF/LJh1tMl/+NynYPjUPwJW8A0z9XOo1N6/xZxPN7Qe4CTQB1Tmvw0RKWoilSEeyY5iffpotlBbU3xq23LEYFVtBIvYnI13qu2gzvmciPHbBCPfNuB+4Dfd/cVSFyYi5WOyF0Rt9NyYrbBDWa5V9TWsbzk3uh1NpNgQjbC6oWbWfg6V3Pu3mPk8hLsY+LS7P1PqYkSkPBXrBTGT2lhkqs1k7vK0f/fLW+YM0Wo7qHPOKQh3/4zCV6S63b3jMj71/rdSH4+SygShWBczohZMRxjBx6hBc12MXbdesaDj6iu5928x8xkBi0iVKfYg7O4dl+XtZpvtTLeFdi+r5N6/xSzkSCIRqWDzfRB2102biUejrG+p420XNLO+pY54NLqosLx5S/uCR88r0by7oZULbUUWWVqFo92BkYm84+YheIjW3lw37fSLye+ttEbpJbDgrcgiUqFyezxMjnaP9Y2wpiHOycHxqamFdU01RR+EVWqj9OVS0ikIM7vFzF42syNm9plZ7rvdzNzMtpWyHhHJt3tfF8l0mpOD47x8aoiTg+MAnB5Okso40YiRyjgnzozTVKvx2lIrWQCbWRT4GvBB4HJgp5ldXuS+ZuBu4LFS1SIixb3SM8TpoURe2KYzObvYJn8BK226ciUo5X/SrgOOuHsXgJl9F7gNKNzE8UfAnwC/X8JaRKSIRCoDBpHs1rbJHW5mEIva1BTE+qZaTo8k2LnnQFVsEV4upZyC2AAcz3ndnb02xcyuBjrc/QezvZGZ3WlmB83sYG9v79JXKlKl4tEgcTMZx93JZDxY0+uwua2JLetXsbmtiUQ6w+BYkqePD3Dq7DhPHx/g9+9/tmK3CC+XUgZwsad+U3+HMbMI8KfA7831Ru6+x923ufu2tra2JSxRpLpddsEq1jbWEIsaaXdiUWN1Q4x4LJLXt+H0cIJ02vEMRM3wDJwZTfLFB18K+7ewopVyCqIb6Mh5vRF4I+d1M/AO4BEL/t6zHthrZre6u9aZiSyDyY0P61tieRsfbr6shZ8c6p06UiidDsI5Ejk3VeEZ52hf8S3C8+1oVk2dz4op5Qj4CeBSM+s0sxrgDmDv5BfdfdDd17n7JnffBBwAFL4iy6jYxofbr9nAk68P0tZcy9vXN9PWXEsGSKSdsWR66ldqhoYQ893IUW2dz4op2QjY3VNm9kmCZu5R4Bvu/oKZ7QIOuvve2d9BRJZD4VrenXsOTOtIFjVIF+RtxoO+D4Xm29Gs2jqfFVPShX3u/gDwQMG1e2a49+ZS1iIixRVOAxw+dZam2hhdvcNTqyAKw3fSQJGVEfPtaFZtnc+K0cpqkSpWbCfc2fEUAyNJ4rHI1NrgmaSdaVMITdnewblbmceSaZpqY3lhPdN9ldr5rBgFsEgVm9wJ1zd8rqNZxj2Y801lyB7rNqvCKQQzI5nO5HU0mzzCPpHOnAv67DWgKjqfFaNuaCJV7JWeIU4NTjCaDb/RZHrqYM2FqI9HGZ5ITXuwt7axhpb6OA01McyCed9V9XHammqrpvNZMRoBi1SxkYkUGcjZbnzua3Xx6NTnMx0/FC8Ywk1OIRQ+2LvxSw8Xne8dHEvy4KdvWsxvYUXTCFikio0nZx7uZtxxnIx7cPIFUBePEI8adfEIq2qjtDTUzHnQJlTfSRfzpRGwSBWbrb3OROpcODfVRulobZh2qCZQtB9w4cqK7ZvXcP9TJ6rmpIv5UgCLyJyGJ9JsWd/En95xzbSvFc7ZFltZcf9TJ7j9mg3s7+pX8/YcCmARKSpiwZzwZIe0H73YM69uaDNtsNjf1T/tRI1qpwAWqSKFUwOzqY2dewiXSqcZTaSnrfndxfQRsDZYzJ8ewolUiWK9F2YzkUoznkwzkUqTDNoG552ckUyn2b2va9r36YHb/CmARarE7n1dJFLTjx+aScaDh3STG+Ec8k7OOD2U4JWeoWnfd9dNm4M1xfNYHVHtFMAiVeLwqbP0jSRIpZ2oGamZGjzMImKGYcEJGpY9UaNAtR0tvxiaAxapEsm0k844aTzv4dr5yGQ86AWcze6aaPE30WnJ86MAFqkS7k5uX52FnLGZe05cc12cznVNS1dgFVIAi1QJM8OYffPFXNa31OVtpNi+eY0O6lwEzQGLVAl3X1T4RoxpJ2fc/9SJqj7RYrE0AhapEhNFHpidjzWNNXkbKYqdnFFtJ1oslgJYpEokFrDqIVfh6ReHT53lwpb6vHu04eL8aApCROal8PSL4Yk0p4cn8u7RhovzowAWkXnLbai+pjHOwGhSGy4WQVMQIjK1OmK2VRKFS37XNtaSTDvtzXUL7nBW2Jui2lZRKIBFqsRs4eoFH+djLJnm0vbmBXc4K9a2cqYGP5VKUxAiVSJynjvfItlTMCIWBEXaWdLphty2lZPTGvGoFW3wU6k0AhapErMtgrDJf/i5UXBuO8p0JkMq4xzrGyWdbchz69b1ixqpqm2lAlikYp1P71+f+sc5uX0fUplgG3M8ArGYkXHY+9xJOtcd5u4dly2ovo7WBnqGxqfWEUP1raLQFIRIBTrf3r/FTKQzjKcyTKQzZDyYiohFIkQskv0I9z16dME1qm2lRsAiFWn3vi76RyYYnih+nPxCZApGyBGDkcTC3//mLe3sovihntVCASxSgZ7tPsPoIsJxPjIOjTXRuW+cRbW3rVQAi1SAwvneUoVvKpMhYkH4Zhw+fmNnSX5OtVAAi6xwxdbTlkp9PMpIIk1jTZSP39i54AdwElAAi6xwu/d1kUyn6RtOTTVLL5XnvvBPSvbe1UgBLLLCvdIzxOBokkjEpg7MLIVVdYqLpaZ/oyIrXCIVnBkfyR7ytpCz3uYSMfjAljadfrHEFMAiK1w8aowlpx+YOV8GNNfFpuZ2P7CljZ+81MNwIj21/rcuFuEnL/WQSDupTIbTQxP86/uf5dduuJj9Xf0K5QUyX8jJfCHatm2bHzx4MOwyRMrGzj0HeOnNQc6Op6YCczGzEE01ES5oqc/bofbyybMk005NLHJud1w6QyRiXLSmIe+cOB1BX1TRv5doJ5zICrd985qp8IXFhS/AcCLDGwMjedcS6eA8uYgZhhExI+3BUffV3ExnsUoawGZ2i5m9bGZHzOwzRb7+r8zsRTN7zsx+YmYXl7IekUr04PMnFx26hcZS068VG8IVXqu2ZjqLVbIANrMo8DXgg8DlwE4zu7zgtqeBbe6+Fbgf+JNS1SNSqV46OVSS983t0RCPBkfaZzKOu5PJJn6sIEGqrZnOYpVyBHwdcMTdu9w9AXwXuC33Bnf/qbtP/ufyALCxhPWIyHnoHZrgpZND9A5N8CvvXM+aphosAml3LAKraqO0NNRUdTOdxSrlKogNwPGc193A9bPc/zHgwWJfMLM7gTsBLrrooqWqT0Rm0dZcy0XZh2tPvj7Ir2dXPOQ2zoHqbqazWKUM4GJTRkVnqszso8A24L3Fvu7ue4A9EKyCWKoCRWRmk6sgGmpijCZS7O/qL3r8kAJ34UoZwN1AR87rjcAbhTeZ2Q7gc8B73X2i8Osics6Nf/wQ3YPn/phsbKldlp8708O1lXCoprsznsxgBnXxxXVvW2qlDOAngEvNrBM4AdwBfCT3BjO7GtgN3OLuPSWsRaSszSfICsMXmPa6VIo9XCvnQzXHk2nGk2nGkmnGkxlODo7xau8wV3W08o4NLaHWlqtkAezuKTP7JPAjIAp8w91fMLNdwEF33wt8GWgC/sqC/ZOvu/utpapJpBzNN8iWK2wheDo/mkjlbbAofLiWe6gmnJuq2L2va9kDOJHKZMM2zehEiu4zYzzXPchz3Wd49vggJ88GHeI+fmNndQQwgLs/ADxQcO2enM93lPLni6wE5RRkk+KxCL1DE3mtJwtrCfNQzWQ6MzXCHZtIc/T0CM+dCML22e4znB5OTPuexpoo6TLb+ateECIhmynIXukZymt+U0q5T8wdmEhl2NhaP7UK4v6nTrB14+q8EF7OQzXTGQ/CNpFmNJHilVNDPNc9yLPZUe7AaHLa9zTXxXjnhha2bmzhyo2rubazlXVNdUte22IogEVC1tHawNHTwwyNn+vnWxMzhifSPH18gHTGOT1c2umHYuPCo6dHpnpLrKqLTRuR33XTZu7Z+8KcUxULkck446kgcIcnUrx8cohnjp/h2e4z/KI76HtRqKU+ng3bIHAvaW+ioSZGXTxCbSxKXbz8Oi8ogEVCtn3zGh4/1k/EgrBLpDOMJIJITKbzPy6n3N4SZ8ZSPH9iIO/rS3mo5uRKhbFkmqGJJC+cGJyaTnj+xFmGJ6YHbmtDnCs3rubKjha2blzNW9ubqK+JUhePUhsLQrfcKYBFQra/q5+WuhhnxpIksyPOcpDbV9gdRpPT/yOwmEM1J1cqnB1PBmGbHeE+f+IsY8npZ9q1NdVOhe1VHavZ3NZIffxc4MZKeBJIqSiARUL2Ss8QQ+Mp4tFzrR4nUpllraE292ensz/bCSaHs7m72Na1kysVBscSPPXaGZ45PsCz3YO8+MbZor/fC1vqpuZvr7loNRevbcwb4VopOs8vMwWwyDIrXPM7MpEq+YkWc4lFbWr+eSI7+JwMZMtOjdTGoud1IkYqHQRu/0iCJ48N8NTxAZ49PsihbG/hQhtb66emFK65qJWL1jZQF49SF4tSU9j1p0IogEWWUbE1vxOpTLbBeXpq0LmcDFjfUjf1IO308ARnx1JEIzbV3D2dcaIGPUPjM65Vnlyp0Ds0wcFj/Tz1ehC4L58aIl2kX+amtQ3ZwF3Ntk2tbFhdv6KnExZCASyyjIqt+Y1FjETalz14JxlwrG80CNmIcevW9XSua+K+R49OrQNeVRultiaWV/fIRJKvP/Iqm9Y18NjRfp58LZhSeOXU0LT+xAZsbmvkyo2rufqi1Vy7aQ3rW+oqajphIRTAIsvo+MAoiWQ6b4nXJC/4uFwyQNwgFjMyDnufO8mn3v/WvCPob/zSw9TFImTcSaTSjEwE63Ff7x/jfV/52bSaIwaXtjezdWMwnXBtZyvtq+oqejphIRTAIiVUON87nkhxeuTcpoESnSB/3mKRIBQjBqlMhvsePcrdOy5jPJmme2CUeMR4rX+URMpJpKc/MItGjLdd0MyVHS286+I1XLuplXVNtdTFo0TLZVlHGVIAi5RIsfne3PAtR+4O7gyNp/jkXz7FM8fP0D0wVvTemqjxS5e2cdtVG7h2U0Tv5HIAABD4SURBVCutjTVVPZ2wEApgkRLZva+LvuFxRhLLu6RsIZLpDBn3vBH5D557c+rz2liEjtZ6RibSJNIZNq1t4Lffewk7rlgfQrWVQwEsUiJPv97PeKpM5hjmkCqYC4lFjKsvWs27Lm7luk1ruPqiVlbVxzWdsMQUwCIlslLCN1dN1PiVrRfyh7e9g6bamKYTSkwBLFLF1jXG+fpH38U7N6ymvqb8eydUGgWwyBLZufvn7D86MPeNIfubT7ybyy9sIa7lYKFTAIssgZUSvvEIfPHBlxd0htu9Dx3O25zx8Rs7uXvHZSWuuLLpP4EiS2AlhG9tFCKRyLTtxI8cmvs4xnsfOsxXHz7CWDJNLBI0Xv/qw0e496HDy1B55dIIWKSCXZJt2TiWTNM9MEZ9PMLJwfGpxjvNdTG+9MNDcx4Iet+jR4nYzBs2ZGEUwCILULjDrVy1N9dNNUs/dXaMkUSaCEbUjFTa6R2e4NTZCTa3Nc56IOhIIhj55opYcF0WTgEscp4eOdTDb3zzianXM+0UKxeTi+EmdxBHIufaXqaTQZDOdSBoY00wis5dBpzx4LosnAJYZA6/+92n2PvcyaluYcVaK5ar3PneiVQGHCLmU71+gWlrfYudbPzxGzv56sNHSGUyUy0qMx5cl4VTAIvM4ne/+xR//cy5LbkrKXyBvPneqIFFLNv+Mrjm0SB8u3qH8+aFO9c15b3P5DyvVkEsLQWwyCxyw3clSqV9ar43A2TSTsea6c3XnczUgaC9wwk+ct2aae91947LFLhLTMvQRCpYMpNhIpUhmQmmH+prorQ31zE4lqS9uY4LV9WxvqWWmmiEjENNNEJ7cw37u/rDLr0qaAQsklW4suHOX1r585vu5z7m9mSbnEjpHZ7gwpZ61jXV5XyPT5sDltJQAIsQhO/v/dUzDE2kSKWdN86MceBoX9hlLVruwZpGsIEi98Hc8EQwDdHWfC6Ax5JpNpbx0rpKogCWqlS4dThC/ghxpaiPGWOzdF2LR84dN59IZzDyl5ytaYzTP5KksTY2NS+cTDt33bR5yWst/BvG+WyDrlSaA5aqkc44z58Y5L1/8vC0rcMrKXzrYhFikeBjU1181ntjUSPtTiy72qGw49naxlqa62J588K7br1iyYNx8nSQhWyDrmQaAUvFSqYzPH9ikMeO9vNYVx9PHBtgeCIVdlmLdukFzVOfjyZS9A4nZrw397j57oEx4lGbtuTs0vZmvnPnDSWtudhp0MU2fFQbBbBUjEQqw3PdZ3jsaD8Huvp46rWBitsqW9gevT4+8060iOVvRd66YRV7nztJxJhzydlSOz4wyur6/NF6sQ0f1UYBLCvWeDLN06+f4bGjfTx+tJ8nXxsIdntVsMIDKsaSadY11XC6yCj4tisv5E/vuGbq9c49B2hvruHsWGpqBLyqPsb+rn7uLnHdHa0N9AyNT42AJ2uv9od9CmBZMUYTKZ58bYDHuvp5/Gg/jx+rvrWqGQ/+PeQ+MPvK7Vey+2dH8ua1t3e25oUvBKPQtY21oSw5u+umzdyz94VptZfiYd9KogCWsrPpM38bdgllI5KzjAygLh7Nm1aYDLATgxN5rSdPDE7wyKGevPnVMEehN29pZxfBXHBu7dU8/wsKYCkBBejSyV1GlsFpiEemPTDbuecAiVSavuFU3sO1wgdcYY9Cb97SXvWBW0gBXAEKu3XdunU9L715lkOnRqbu2XJBIz/83ZsVjitMLGo5oRqf1iQH4PCps5wdT+X1+e0bSZBKn827T6PQ8lPSADazW4CvAlHgPnf/YsHXa4G/AN4F9AG/6u7HSlnT+SpcsL+9s5XDp4boGz23nGltQ7CIvXtwYuraxpZaHv3sjmUPvHTGizaQOXRqROG7wjTXRmlfVTfnaDWZDjZi5Pb5zWScRHr6Bg2NQstLyQLYzKLA14B/BHQDT5jZXnd/Mee2jwED7v5WM7sD+BLwq6Wq6XwVO2ix2NlfQRjnry/tHpxQ4Mm8GdBcF8tr9bh14+p5jVZrYhHGEmkyntPn14PrUt7MvTT9Tc1sO/AFd/8n2defBXD3P86550fZe/abWQw4CbT5LEVt27bNDx48eF613Pabv8+Jo0fO+/dwdjx53t8jMh+F63kBmufY1TaT0USKjAcrGqbe3yzvpAtZvPq2DnZ85Lf4/P9xxUK+vdj/5CXdirwBOJ7zujt7reg97p4CBoG1hW9kZnea2UEzO9jb21uickXC4UBNbOFH+0x+r5kRidjUCReLeU9ZHqX8z2OxxC8c2c7nHtx9D7AHghHw+RbyN3/+lfP9FkBP82VxNrbU5j0XmLS9s5UX3hxa0pMlJhvd6OHaylLKAO4GOnJebwTemOGe7uwURAtQNqvrt3e2Fp3zlerxz666kH2He6c9dJ1IZRhOnNt111QT4fldH5z2/fc+dHhZjvHRw7WVqZRzwDHgMPAB4ATwBPARd38h555PAO9099/KPoT75+7+4dnedyFzwIux0lZBzObYF3+5rOop5nxqjBnkdmKMGVy7qXXa/17AtGsDo4lpy/TefuGqacv5CneTiSxQ0TngkgUwgJl9CPhPBMvQvuHu/97MdgEH3X2vmdUB3wKuJhj53uHuXbO953IHcKVKpDL84sQgjx3t48CrfTz5+gAjE9Mb16xfVccNm9dw/ea1XN+5hs51jdNO0RWROS1/AJeCAnhhxpNpnj0edArb/2ofTx8fYDw5vXHNxtb6IHA713J951o61tQrcEUWr+gfIq1RqVCjiRRPv36GA6/2sb+rj+e6B0mkpwfuprUN3LB5LddnQ/ctq+tDqFakOimAK8TwRNApbP+rp9nf1c/zJwZJZ6b/7eaStkau71zL9kuCKYX2VXVF3k1EloMCeIUaHEty8Fg///DqaR7r6ufQm0OkC6aTDLj0giau37yWd29ey3Wda1jbVBtOwSIyjQJ4hegfSfBYVx8/fzVoPn741NC0BdMRg7dfuIrrOtfw7kvWcd2mNbQ0LGx3lYiUngK4TPUMjbP/1T72v9rH48f66eodmXZPNGJc8ZZVXN+5hne/dR3bLm5d8HZWEVl+CuAy8ebgGI++cpoDXX0cPDbAa/3TTymIR413bmjhus61vOeta3nXxa3a6y+ygulPbwjcndf7Rvn5q31B4L7Wz4kz49Puq41F2Lqxhes613DjJeu4+uJW6mY5hFFEVhYF8DJwd17tGebRV/t4rKuPJ18boGdoeo+A+niUKztWZ6cU1nJ1R6taCopUMAVwCbg7L755lp+/2scT2dN6+0amn1rbWBPlqotWc13nGt5zyTqu6lhNLKrAFakWCuAlkEpl+MUbg/z81T4OHuvn6dfPcGZsei/hVXUxrr6olWs3reE9b13L1o2riUa0y0ykWimAF2A8keKZ7sGpB2bPdp9haDw17b7WhjhXX9TKdZta2X7JOt65oWXq2BgREQXwPCTTGR4/2s/PXz3NwWMD/OLEIKOJ6Y1r1jXVnBvhXrKWt1/YTCSiKQURKU4BPIeh8STX/4efFA3cC1bVTo1w3/3WdVza1kRUc7giMk8K4Dk018W5sKWOV3tH2LC6nqs6VnNtNnAvaWvSHK6ILJgCeB7+44evoqU+xobWBuIa4YrIElEAz8OVHavDLkFEKpCGcyIiIVEAi4iERAEsIhISBbCISEgUwCIiIVEAi4iERAEsIhISBbCISEgUwCIiIVEAi4iERAEsIhISBbCISEjM3cOu4byYWS/w2jL+yHXA6WX8ebNRLTMrp3pUy8zKqZ7lrOW0u99SeHHFBfByM7OD7r4t7DpAtcymnOpRLTMrp3rKoRZNQYiIhEQBLCISEgXw3PaEXUAO1TKzcqpHtcysnOoJvRbNAYuIhEQjYBGRkCiARURCogCeBzP7IzN7zsyeMbMfm9lbQqzly2Z2KFvPX5tZaCeGmtm/MLMXzCxjZqEs5zGzW8zsZTM7YmafCaOGnFq+YWY9ZvZ8mHVka+kws5+a2UvZ/40+FWItdWb2uJk9m63lD8OqJaemqJk9bWY/CLMOBfD8fNndt7r7VcAPgHtCrOXvgHe4+1bgMPDZEGt5HvjnwL4wfriZRYGvAR8ELgd2mtnlYdSS9U1g2mL7kKSA33P3twM3AJ8I8d/NBPB+d78SuAq4xcxuCKmWSZ8CXgq5BgXwfLj72ZyXjUBoTy7d/cfunsq+PABsDLGWl9z95bB+PnAdcMTdu9w9AXwXuC2sYtx9H9Af1s/P5e5vuvtT2c+HCMJmQ0i1uLsPZ1/Gs79C+zNkZhuBXwbuC6uGSQrgeTKzf29mx4F/Sbgj4Fz/N/Bg2EWEaANwPOd1NyGFTDkzs03A1cBjIdYQNbNngB7g79w9tFqA/wT8GyATYg2AAniKmT1kZs8X+XUbgLt/zt07gG8Dnwyzluw9nyP4a+a3w64lRFbkmtZV5jCzJuD/Az5d8De5ZeXu6ewU3kbgOjN7Rxh1mNmvAD3u/mQYP79QLOwCyoW775jnrX8J/C3w+bBqMbP/E/gV4ANe4oXc5/HvJQzdQEfO643AGyHVUnbMLE4Qvt929/8Zdj0A7n7GzB4hmCsP42Hle4BbzexDQB2wysz+h7t/NIRaNAKeDzO7NOflrcChEGu5Bfi3wK3uPhpWHWXiCeBSM+s0sxrgDmBvyDWVBTMz4L8BL7n7fwy5lrbJ1TpmVg/sIKQ/Q+7+WXff6O6bCP7/8nBY4QsK4Pn6Yvav3c8B/5jgCWpY/gvQDPxddlncfw2rEDP7Z2bWDWwH/tbMfrScPz/7MPKTwI8IHjJ9391fWM4acpnZd4D9wNvMrNvMPhZWLQQjvV8D3p/9/8kz2VFfGC4Efpr98/MEwRxwqMu/yoW2IouIhEQjYBGRkCiARURCogAWEQmJAlhEJCQKYBGRkCiARURCogCWFcPMPm1mDUt133Iys2+a2e3Zzx8Jq32nlBcFsKwknwbmE6zzvW9JZdtjisybAljKkpk1mtnfZpt4P29mnwfeQrCj6qfZe/7MzA7mNvk2s7sL75vh/Xea2S+y7/2l7LXfNrM/ybnnN8zsP2c//2i2qfgzZrZ7MmzNbNjMdpnZY8B2M7vHzJ7Ivu+e7JZgkaIUwFKubgHecPcr3f0dBC0E3wDe5+7vy97zOXffBmwF3mtmW9393iL35bHgRJMvAe8naBB+rZn9U+B+ggbzk34V+J6ZvT37+XuyHb3SBG1JIegP/by7X+/ujwL/xd2vzdZcT9A0SaQoBbCUq18AO8zsS2b2S+4+WOSeD5vZU8DTwBUEp2LMx7XAI+7em+0n8W3gJnfvBbrM7AYzWwu8DfgH4APAu4Ansj1tPwBszr5XmqDj2KT3mdljZvYLgoC/4nx+01Jd1I5SypK7HzazdwEfAv7YzH6c+3Uz6wR+H7jW3QfM7JsE7QXnY7Zpge8BHybo1vXX7u7ZaYT/7u7Fjn8ad/d0tqY64OvANnc/bmZfOI+apAppBCxlKTtNMOru/wP4CnANMETQCQ5gFTACDJrZBQTnwk3Kva+YxwimLNZl53J3Aj/Lfu1/Av80e+172Ws/AW43s/ZsbWvM7OIi7zsZtqezjdBvn+/vV6qTRsBSrt4JfNnMMkAS+G2CtpcPmtmb7v4+M3saeAHoIpgqmLQn977CN3b3N83ss8BPCUbDD7j732S/NmBmLwKXu/vj2WsvmtkfAD82s0i2nk8ArxW87xkz+3OC6ZNjBK0XRWakdpQiIiHRFISISEg0BSEVLbs+t7bg8q+5+y/CqEckl6YgRERCoikIEZGQKIBFREKiABYRCYkCWEQkJP8/cfGS4tYLiAkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 연속형변수 확인\n",
    "# 'age', 'stat_overall', 'stat_potential'\n",
    "# 스케일링\n",
    "ind_idx_num = ['age', 'stat_overall', 'stat_potential']\n",
    "de_idx = \"value\"\n",
    "scaler = StandardScaler()\n",
    "comdata[idx_num] = scaler.fit_transform(comdata[idx_num])\n",
    "\n",
    "\n",
    "plt.figure(figsize  = (8, 6))\n",
    "for i in ind_idx_num:\n",
    "    sns.lmplot(x = i, y = \"value\", data = comdata)\n",
    "    sns.distplot(comdata[i], fit = norm)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(12760, 24)\n"
     ]
    }
   ],
   "source": [
    "# One-Hot-encoding 더미변수 생성\n",
    "comdata_dummies = pd.get_dummies(comdata.iloc[:, 2:])\n",
    "print(comdata_dummies.shape) # (12760, 24)\n",
    "\n",
    "# 데이터셋 분리\n",
    "tr_data = comdata_dummies.iloc[:8932]\n",
    "te_data = comdata_dummies.iloc[8932:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 독립변수, 종속변수 분리\n",
    "col = tr_data.columns\n",
    "y_col = \"value\"\n",
    "x_col = [i for i in col if i != y_col]\n",
    "\n",
    "# train /test set split\n",
    "x_train, x_test, y_train, y_test = train_test_split(tr_data[x_col], tr_data[y_col], test_size = 0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 기본 pipe 생성\n",
    "pipe = Pipeline([(\"regressor\", LinearRegression())])\n",
    "\n",
    "# paramgrid\n",
    "param_grid = [{\"regressor\":[LinearRegression()]},\n",
    "             {\"regressor\":[RandomForestRegressor()], \"regressor__n_estimators\":[100,200,300,400],\n",
    "             \"regressor__max_depth\":[3,6,8,10], \"regressor__min_samples_split\":[2,3,4,5], \"regressor__min_samples_leaf\": [1,3,5,7]},\n",
    "             {\"regressor\":[XGBRegressor()], \"regressor__colsample_bylevel\":[0.6,0.8,1], \"regressor__learning_rate\":[0.01,0.1],\n",
    "             \"regressor__max_depth\":[3,6,8,10], \"regressor__min_child_weight\":[1,3], \"regressor__n_estimators\":[100,200,300,400]}]\n",
    "\n",
    "\n",
    "# eval_set = [(x_test, y_test)], eval_metric = \"rmse\", verbose = True\n",
    "# eval_set : ensamble 모형에서 사용\n",
    "gs = GridSearchCV(pipe, param_grid, cv = 5, n_jobs=-1)\n",
    "\n",
    "model = gs.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model score & best_parameter\n",
    "print(model.best_score_)\n",
    "print(model.best_estimator_)\n",
    "y_pred = model.predict(x_test)\n",
    "y_true = y_test\n",
    "print(mean_squared_error(y_true, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 중요변수 확인 : \n",
    "model.best_estimator_.named_steps[\"regressor\"].feature_importances_\n",
    "plt.figure(figsize = (20,5))\n",
    "sns.barplot(x = list(range(23)), y = model.best_estimator_.named_steps[\"regressor\"].feature_importances_)\n",
    "plt.xticks(list(range(23)), x_train.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 실제 데이터 예측\n",
    "rel_y_pred = model.predict(te_data[x_col])\n",
    "te_data\n",
    "\n",
    "submit = pd.read_csv(\"c:/itwill/4_python-ii/data/submission.csv\")\n",
    "submit.info()\n",
    "submit[\"value\"] = rel_y_pred\n",
    "submit.head()\n",
    "submit.to_csv(\"c:/itwill/4_python-ii/data/submission.csv\", index = None, encoding = \"utf-8\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lgb = LGBMRegressor()\n",
    "train_ds = lgb.Dataset(x_train, label = y_train) \n",
    "test_ds = lgb.Dataset(x_test, label = y_test) \n",
    "\n",
    "param_grid = {\n",
    "    'n_estimators': [400, 700, 1000],\n",
    "    'colsample_bytree': [0.7, 0.8],\n",
    "    'max_depth': [15,20,25],\n",
    "    'num_leaves': [50, 100, 200],\n",
    "    'reg_alpha': [1.1, 1.2, 1.3],\n",
    "    'reg_lambda': [1.1, 1.2, 1.3],\n",
    "    'min_split_gain': [0.3, 0.4],\n",
    "    'subsample': [0.7, 0.8, 0.9],\n",
    "    'subsample_freq': [20]\n",
    "}\n",
    "\n",
    "model = lgb.train(param_grid, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100)\n",
    "predict_train = model.predict(x_train)\n",
    "predict_test = model.predict(x_test)\n",
    "\n",
    "mse = mean_squared_error(y_test, predict_test)\n",
    "r2 = r2_score(y_test, predict_test)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
