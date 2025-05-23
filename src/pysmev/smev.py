import math
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Sequence, Optional
import statsmodels.api as sm


class SMEV:
    def __init__(
        self,
        min_rain: float,
        separation: int,
        return_period: list[Union[int, float]],
        durations: list[int],
        time_resolution: int,
        min_event_duration: Union[None, int] = None,
        left_censoring: Union[None, list] = None,
        tolerance=0.1,
    ):
        """Initiates SMEV class.

        Args:
            min_rain (float): Minimum precipitation value to consider as a storm.
            separation (int): Separation time between independent storms [min]
            return_period (list[Union[int, float]]): List of return periods of interest [years].
            durations (list[Union[int]]): List of durations of interest [min].
            time_resolution (int): Temporal resolution of the precipitation data [min].
            min_duration (Union[None, int], optional): Minimum duration of storm [min]. Defaults to None.\
                If None, it is set to 0.
            left_censoring (Union[None, list], optional): 2-elements list with the limits in probability \
                of the data to be used for the parameters estimation. Defaults to None.\
                If None, it is set to [0, 1].
        """
        self.min_rain = min_rain
        self.separation = separation
        self.return_period = return_period
        self.durations = durations
        self.time_resolution = time_resolution
        self.min_event_duration = min_event_duration if min_event_duration is not None else 0
        self.left_censoring = left_censoring if left_censoring is not None else [0, 1]
        self.tolerance = tolerance
        self.__incomplete_years_removed__ = False
        
    def remove_incomplete_years(
        self, data_pr: pd.DataFrame, name_col="value", nan_to_zero=True
    ) -> pd.DataFrame:
        """Function that deletes incomplete years in precipitation data.
        An incomplete year is defined as a year where observations are missing above a given threshold.

        Args:
            data_pr (pd.DataFrame): Dataframe containing (hourly) precipitation values.
            name_col (str, optional): Column name in `data_pr` with precipitation values. Defaults to "value".
            nan_to_zero (bool, optional): Set `nan` to zero. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe containing (hourly) precipitation values with incomplete years removed.
        """
        # Step 1: get resolution of dataset (MUST BE SAME in whole dataset!!!)
        time_res = (data_pr.index[-1] - data_pr.index[-2]).total_seconds() / 60
        # Step 2: Resample by year and count total and NaN values
        yearly_valid = data_pr.resample("YE").apply(
            lambda x: x.notna().sum()
        )  # Count not NaNs per year
        # Step 3: Estimate expected lenght of yearly timeseries
        expected = pd.DataFrame(index=yearly_valid.index)
        expected["Total"] = 1440 / time_res * 365 # 1440 stands for the number of minutes in a day
        # Step 4: Calculate percentage of missing data per year by aligning the dimensions
        valid_percentage = yearly_valid[name_col] / expected["Total"]
        # Step 3: Filter out years where more than 10% of the values are NaN
        years_to_remove = valid_percentage[valid_percentage < 1 - self.tolerance].index
        # Step 4: Remove data for those years from the original DataFrame
        data_cleanded = data_pr[~data_pr.index.year.isin(years_to_remove.year)]
        # Replace NaN values with 0 in the specific column
        if nan_to_zero:
            data_cleanded.loc[:, name_col] = data_cleanded[name_col].fillna(0)

        self.time_resolution = time_res

        self.__incomplete_years_removed__ = True

        return data_cleanded
        
    def get_ordinary_events(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        dates: list,
        name_col="value",
        check_gaps=True,
    ) -> list:
        """Function that extracts ordinary precipitation events out of the entire data.

        ..todo::
            Bit clumsy function, maybe better to split it into smaller functions.
            Also, is support for both pd.DataFrame and np.ndarray necessary?

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Data with precipitation values.
            dates (list): List with dates of precipitation values.\
                Only relevant if `data` is an array or if `check_gaps==True`.
            name_col (str, optional): Column name in `data` for precipitation values.\
                Only relevant if `data` is a dataframe. Defaults to "value".
            check_gaps (bool, optional): Check for gaps in precipitation time series. \
                Defaults to True.

        Returns:
            list: Consecutive values above `self.min_rain` separated by more `self.seperation`.
        """
        if isinstance(data, pd.DataFrame):
            # Find values above threshold
            above_threshold = data[data[name_col] > self.min_rain]
            # Find consecutive values above threshold separated by more than 24 observations
            consecutive_values = []
            temp = []
            for index, _ in above_threshold.iterrows():
                if not temp:
                    temp.append(index)
                else:
                    if index - temp[-1] > pd.Timedelta(hours=self.separation):
                        if len(temp) >= 1:
                            consecutive_values.append(temp)
                        temp = []
                    temp.append(index)
            if len(temp) >= 1:
                consecutive_values.append(temp)
        elif isinstance(data, np.ndarray):
            # Assuming data is your numpy array
            # Assuming name_col is the index for comparing threshold
            # Assuming threshold is the value above which you want to filter

            above_threshold_indices = np.where(data > self.min_rain)[0]

            # Find consecutive values above threshold separated by more than 24 observations
            consecutive_values = []
            temp = []
            for index in above_threshold_indices:
                if not temp:
                    temp.append(index)
                else:
                    # numpy delta is in nanoseconds, it  might be better to do dates[index] - dates[temp[-1]]).item() / np.timedelta64(1, 'm')
                    if (
                        (dates[index] - dates[temp[-1]]).item()
                        > (self.separation * 3.6e12)
                    ):  # Assuming 24 is the number of hours, nanoseconds * 3.6e+12 = hours
                        if len(temp) >= 1:
                            consecutive_values.append(dates[temp])
                        temp = []
                    temp.append(index)
            if len(temp) >= 1:
                consecutive_values.append(dates[temp])

        if check_gaps:
            # remove event that starts before dataset starts in regard of separation time
            if (consecutive_values[0][0] - dates[0]).item() < (
                self.separation * 3.6e12
            ):  # this numpy dt, so still in nanoseconds
                consecutive_values.pop(0)
            else:
                pass

            # remove event that ends before dataset ends in regard of separation time
            if (dates[-1] - consecutive_values[-1][-1]).item() < (
                self.separation * 3.6e12
            ):  # this numpy dt, so still in nanoseconds
                consecutive_values.pop()
            else:
                pass

            # Locate OE that ends before gaps in data starts.
            # Calculate the differences between consecutive elements
            time_diffs = np.diff(dates)
            # difference of first element is time resolution
            time_res = time_diffs[0]
            # Identify gaps (where the difference is greater than 1 hour)
            gap_indices_end = np.where(
                time_diffs > np.timedelta64(int(self.separation * 3.6e12), "ns")
            )[0]
            # extend by another index in gap cause we need to check if there is OE there too
            gap_indices_start = gap_indices_end + 1

            match_info = []
            for gap_idx in gap_indices_end:
                end_date = dates[gap_idx]
                start_date = end_date - np.timedelta64(
                    int(self.separation * 3.6e12), "ns"
                )
                # Creating an array from start_date to end_date in hourly intervals
                temp_date_array = np.arange(start_date, end_date, time_res)

                # Checking for matching indices in consecutive_values
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for gap_idx in gap_indices_start:
                start_date = dates[gap_idx]
                end_date = start_date + np.timedelta64(
                    int(self.separation * 3.6e12), "ns"
                )
                # Creating an array from start_date to end_date in hourly intervals
                temp_date_array = np.arange(start_date, end_date, time_res)

                # Checking for matching indices in consecutive_values
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for del_index in sorted(match_info, reverse=True):
                del consecutive_values[del_index]

        return consecutive_values

    def remove_short(
        self, list_ordinary: list
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
        """Function that removes ordinary events too short.
        Also, it calculates the mean number of ordinary events per year after removal.

        Args:
            list_ordinary (list): List with all ordinary events.
            min_duration (int): Minimum duration of storm [min].

        Returns:
            np.ndarray: Array with values of ordinary events.
            np.ndarray: Array with dates of ordinary events.
            pd.DataFrame: Dataframe with number of ordinary events per year.
            float: Mean number of ordinary events per year.
        """
        if isinstance(list_ordinary[0][0], pd.Timestamp):
            ll_short = [
                True
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=self.min_event_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (
                    ev[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    ev[0].strftime("%Y-%m-%d %H:%M:%S"),
                )
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=self.min_event_duration)
                else (np.nan, np.nan)
                for ev in list_ordinary
            ]

            arr_vals = np.array(ll_short)[ll_short]
            arr_dates = np.array(ll_dates)[ll_short]

            filtered_list = [x for x, keep in zip(list_ordinary, ll_short) if keep]
            list_year = pd.DataFrame(
                [filtered_list[_][0].year for _ in range(len(filtered_list))],
                columns=["year"],
            )
            n_ordinary_per_year = list_year.reset_index().groupby(["year"]).count()
            n_ordinary = (
                n_ordinary_per_year.values.mean()
            )  # ordinary events mean value (one of input to smev)
        elif isinstance(list_ordinary[0][0], np.datetime64):
            ll_short = [
                True
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=self.min_event_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (ev[-1], ev[0])
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=self.min_event_duration)
                else (np.nan, np.nan)
                for ev in list_ordinary
            ]

            arr_vals = np.array(ll_short)[ll_short]
            arr_dates = np.array(ll_dates)[ll_short]

            filtered_list = [x for x, keep in zip(list_ordinary, ll_short) if keep]
            list_year = pd.DataFrame(
                [
                    filtered_list[_][0].astype("datetime64[Y]").item().year
                    for _ in range(len(filtered_list))
                ],
                columns=["year"],
            )
            n_ordinary_per_year = list_year.reset_index().groupby(["year"]).count()
            n_ordinary = (
                n_ordinary_per_year.values.mean()
            )  # ordinary events mean value (one of input to smev)

        return arr_vals, arr_dates, n_ordinary_per_year, n_ordinary
        
    def get_ordinary_events_values(
        self, data: np.ndarray, dates: np.ndarray, arr_dates_oe
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Function that extract ordinary events and annual maximas out of precpitation data. 
        
        Parameters
        ----------
        data (np.ndarray): data of full precipitation dataset
        dates (np.ndarray): time of full precipitation dataset
        arr_dates_oe (np.ndarray): end and start of ordinary event as retruned by remove_short function.

        Returns
        -------
        dict_ordinary (dict): key is duration, value is pd.DataFrame with year, oe_time and value of ordinary event (eg. depth)
            contains ordinary events values per duration.
            example dict_ordinary = {"10" : pd.DataFrame(columns=['year', 'oe_time', 'ordinary'])
        dict_AMS (dict): key is duration, value is pd.DataFrame with year and the annual maxima (AMS) value.
            contains anual maximas for each year per duration.

        """
        dict_ordinary = {}
        dict_AMS = {}
        for d in range(len(self.durations)):
            arr_conv = np.convolve(
                data,
                np.ones(int(self.durations[d] / self.time_resolution), dtype=int),
                "same",
            )

            # Convert time index to numpy array
            time_index = dates.reshape(-1)

            # Use numpy indexing to get the max values efficiently
            ll_vals = []
            ll_dates = []
            for i in range(arr_dates_oe.shape[0]):
                start_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 1])

                end_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 0])

                # Check if start and end times are the same
                if start_time_idx == end_time_idx:
                    ll_val = arr_conv[start_time_idx]
                    ll_date = time_index[start_time_idx]
                else:
                    # the +1 in end_time_index is because then we search by index but we want to includde last as well,
                    # without, it slices eg. end index is 10, without +1 it slices 0 to 9 instead of 0 to 10 (stops 1 before)
                    # get index of ll_val within the sliced array
                    ll_idx_in_slice = np.nanargmax(
                        arr_conv[start_time_idx : end_time_idx + 1]
                    )
                    # adjust the index to refer to the original arr_conv
                    ll_idx_in_arr_conv = start_time_idx + ll_idx_in_slice
                    ll_val = arr_conv[ll_idx_in_arr_conv]
                    ll_date = time_index[ll_idx_in_arr_conv]

                ll_vals.append(ll_val)
                ll_dates.append(ll_date)

            # years  of ordinary events
            ll_yrs = [
                arr_dates_oe[_, 0].astype("datetime64[Y]").item().year
                for _ in range(arr_dates_oe.shape[0])
            ]

            blocks = np.unique(ll_yrs)

            AMS = {}
            for j in blocks:
                indices = [index for index, value in enumerate(ll_yrs) if value == j]
                slice_ll_vals = [ll_vals[i] for i in indices]
                AMS[j] = max(slice_ll_vals)

            df_ams = pd.DataFrame({"year": [*AMS.keys()], "AMS": [*AMS.values()]})
            df_oe = pd.DataFrame(
                {"year": ll_yrs, "oe_time": ll_dates, "ordinary": ll_vals}
            )
            dict_AMS.update({f"{self.durations[d]}": df_ams})
            dict_ordinary.update({f"{self.durations[d]}": df_oe})

        return dict_ordinary, dict_AMS
        
    def estimate_smev_parameters(
        self, 
        ordinary_events: Union[Sequence[float], np.ndarray, pd.Series],  
        data_portion: Optional[Tuple[float, float]] = None
    ) -> list[float]:
        """Function that estimates shape and scale parameters of the Weibull distribution.

        Args:
            ordinary_events: 1D list, array, or series of ordinary event values.
            data_portion (list): Lower and upper limits of the probabilities of data \
                to be used for the parameters estimation.

        Returns:
            list[float]: Shape and scale parameters of the Weibull distribution.
        """
        
        if data_portion is None:
            data_portion = self.left_censoring  # Assumes it's a Tuple[float, float]
        
        # input must be 1D array
        ordinary_events = np.asarray(ordinary_events).flatten()
        sorted_df = np.sort(ordinary_events)
        ECDF = np.arange(1, 1 + len(sorted_df)) / (1 + len(sorted_df))
        fidx = max(1, math.floor((len(sorted_df)) * data_portion[0]))
        tidx = math.ceil(len(sorted_df) * data_portion[1])
        to_use = np.arange(fidx - 1, tidx)
        to_use_array = sorted_df[to_use]

        X = np.log(np.log(1 / (1 - ECDF[to_use])))
        Y = np.log(to_use_array)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        param = results.params

        slope = param[1]
        intercept = param[0]

        shape = 1 / slope
        scale = np.exp(intercept)

        weibull_param = [shape, scale]

        return weibull_param

    def smev_return_values(
        self, return_period: int, shape: float, scale: float, n: float
    ) -> float:
        """Function that calculates return values (here, rainfall intensity) acoording to parameters of the Weibull distribution.

        Args:
            return_period (int): Return period of interest.
            shape (float): Shape parameter value.
            scale (float): Scale parameter value.
            n (float): SMEV parameter `n`.

        Returns:
            float: Rainfall intensity value.
        """

        return_period = np.asarray(return_period)
        quantile = 1 - (1 / return_period)
        if shape == 0 or n == 0:
            intensity = 0
        else:
            intensity = scale * ((-1) * (np.log(1 - quantile ** (1 / n)))) ** (
                1 / shape
            )

        return intensity
    
    def do_smev_all(self, 
        dict_ordinary: Dict[str, pd.DataFrame],
        n: float
        
    ) -> Dict[str, pd.DataFrame]:
        
        dict_smev_outputs = {}
        for d in range(len(self.durations)):
            f"{self.durations[d]}"
            # Example for return levels of 60min duration 
            P = dict_ordinary[ f"{self.durations[d]}"]["ordinary"]

            # Estimate shape and scale parameters of weibull distribution
            # We include S_SMEV.left_censoring but actually it is not needed as it auto reads it from S_SMEV class
            smev_shape, smev_scale = self.estimate_smev_parameters(P)

            # estimate return period (quantiles) with SMEV
            smev_RL = self.smev_return_values(self.return_period,
                                              smev_shape, 
                                              smev_scale, 
                                              n
                                              )
            
            dict_smev_outputs[f"{self.durations[d]}"] = {"SMEV_phat":  [smev_shape, smev_scale],
                                                         "RLs":smev_RL,
                                                         "n" : n
                                                         }

        
        return dict_smev_outputs

    def get_stats(
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Computes statistics of precipitation values.
        Statistics are total percipitation per year, mean precipitation per year,
        standard deviation of precipitation per year, and count of precipitation events per year.

        Args:
            df (pd.DataFrame): Dataframe with precipitation values.

        Returns:
            pd.Series: Total percipitation per year.
            pd.Series: Mean percipitation per year.
            pd.Series: Standard deviation of percipitation per year.
            pd.Series: Count of percipitation events per year.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df is not a pandas dataframe")

        total_prec = df.groupby(df.index.year)["value"].sum()
        mean_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].mean()
        )
        sd_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].std()
        count_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].count()
        )

        return total_prec, mean_prec, sd_prec, count_prec

    def SMEV_bootstrap_uncertainty(
        self, P: np.ndarray, blocks_id: np.ndarray, niter: int, n: float
    ):
        """Function that bootstraps uncertainty of SMEV return values.

        Args:
            P (np.ndarray): Array of precipitation data.
            blocks_id (np.ndarray): Array of block identifiers (e.g., years).
            niter (int): Number of bootstrap iterations.
            n (float): SMEV parameter `n`.

        Returns:
            np.ndarray: Array with bootstrapped return value uncertainty.
        """
        RP = self.return_period

        blocks = np.unique(blocks_id)
        M = len(blocks)
        randy = np.random.randint(0, M, size=(M, niter))

        # Initialize variables
        RL_unc = np.full((niter, len(RP)), np.nan)
        n_err = 0

        # Random sampling iterations
        for ii in range(niter):
            Pr = []
            Bid = []

            # Create bootstrapped data sample and corresponding 'fake' blocks id
            for iy in range(M):
                selected = blocks_id == blocks[randy[iy, ii]]
                Pr.append(P[selected])
                Bid.append(
                    np.full(np.sum(selected), iy + 1)
                )  # MATLAB indexing starts at 1

            # Concatenate the resampled data
            Pr = np.concatenate(Pr)
            Bid = np.concatenate(Bid)

            try:
                # estimate shape and  scale parameters of weibull distribution
                SMEV_shape, SMEV_scale = self.estimate_smev_parameters(
                    Pr, self.left_censoring
                )
                # estimate return period (quantiles) with SMEV
                smev_RP = self.smev_return_values(
                    self.return_period, SMEV_shape, SMEV_scale, n
                )
                # Store results
                RL_unc[ii, :] = smev_RP

            except Exception:
                n_err += 1
        return RL_unc
