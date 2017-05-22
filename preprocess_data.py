def get_fundamental_data(ticker_symbol):
	"""load table from local file"""
	fundamental_data = pd.read_csv('%s_fundamental_data.csv' % (ticker_symbol),index_col=0, parse_dates=True).transpose()
	fundamental_data.index.rename('Date',inplace=True)
	fundamental_data = fundamental_data.reset_index()
	# create index of quarters
	def add_quarter(x):
		if x[0] == '1':
			return x[-2:] + 'q4'
		elif x[0] == '3':
			return x[-2:] + 'q1'
		elif x[0] == '6':
			return x[-2:] + 'q2'
		elif x[0] == '9':
			return x[-2:] + 'q3'
	fundamental_data['quarter'] = fundamental_data['Date'].apply(add_quarter) # e.g 06q4, 08q1
	fundamental_data = fundamental_data.set_index('quarter')
	fundamental_data = fundamental_data.drop('Date',1)
	return fundamental_data
  
def add_release_date(ticker_symbol, fundamental_data):
	release_date = pd.read_csv('release_date/%s.csv'%(ticker_symbol), parse_dates=True)
	fundamental_data = fundamental_data['06q4':] # match the information, since all release date is from 01/01/2007 (q4 2006 data)
	# merge the two tables
	# create common key of quarter
	def add_quarter(x):
		if x[:2] == '1/' or x[:2] == '2/':
			# change to previous year: e.g. if release is in january 07, it should be the q4 data of 06.
			release_year = int(x[-2:])
			previous_year = release_year - 1
			if len(str(previous_year)) == 1: 
				# year is before 2010: e.g. 08,07
				return '0' + str(previous_year) + 'q4'
			else:
				return str(previous_year) + 'q4'
		elif x[:2] == '4/'or x[:2] == '5/':
			return x[-2:] + 'q1'
		elif x[:2] == '7/'or x[:2] == '8/':
			return x[-2:] + 'q2'
		elif x[:2] == '10'or x[:2] == '11':
			return x[-2:] + 'q3'
	release_date['quarter'] = release_date['calendar_date'].apply(add_quarter)
	fundamental_data = release_date.join(fundamental_data, on='quarter')
	return fundamental_data # columns: calendar_date, calendar_time, quarter, and financial indicators....

def merge_data(fundamental_data, market_data, start_date, end_date):
	# for fundamental data: we need to fill the date gap with same data point
	# there are also some na data point will generate for market data since some end of quarter date are weekend.
	trading_days = market_data.index.tolist()
	release_dates = pd.to_datetime(fundamental_data.calendar_date).tolist()
	before_or_after = fundamental_data.calendar_time.tolist()
	for i in range(len(before_or_after)):
		if before_or_after[i] == 'After Market Close':
			for j in range(len(trading_days)-1):
				if release_dates[i] == trading_days[j]:
					release_dates[i] = trading_days[j+1]
					break
	fundamental_data = fundamental_data.set_index(pd.DatetimeIndex(release_dates))
	fundamental_data = fundamental_data.drop('calendar_date',1)
	fundamental_data = fundamental_data.drop('calendar_time',1)
	fundamental_data = fundamental_data.drop('quarter',1)
	merge = market_data.join(fundamental_data, how='left')
	merge = merge.fillna(method='ffill') # use the previous data to fill the na places. forward filling
	merge = merge.dropna()
	return merge
  
  
def prepare_prediction(merged_data, training_test_line, time_interval):
	# first we need to shift tomorrow return to today's data row in order to predict
	merged_data['change'] = merged_data['Close'].pct_change(periods=time_interval)
	merged_data.change = merged_data.change.shift((-1)*time_interval)
	merged_data = merged_data.dropna()
	# then label the prediction data
	merged_data.change[merged_data.change > 0] = 1
	merged_data.change[merged_data.change <= 0] = -1
	features = merged_data.columns[2:-1]
	x = merged_data[features]
	x['Volume'] = merged_data['Volume']
	y = merged_data.change
	# standardize
	standardizer = StandardScaler()
	x_train = standardizer.fit_transform(x[x.index < training_test_line].as_matrix())
	y_train = y[y.index < training_test_line].values
	x_test = standardizer.fit_transform(x[x.index >= training_test_line])
	y_test = y[y.index >= training_test_line].values
	return x_train, y_train, x_test, y_test
