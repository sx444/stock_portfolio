stocks = ['TICKER1', 'TICKER2', 'TICKER3', ...]
start_date = 'year-month-day'
end_date = 'year-month-day'


def get_stocks_returns(stocks, start_date, end_date):
	'''
	get data from yahoo; output is pandas (columns are the stocks, rows are daily returns)
	'''
	returns = pandas_datareader.data.get_data_yahoo(stocks[0], start_date, end_date)[['Close']]
	returns['%s_daily_return'%stocks[0]] = returns['Close'].pct_change(periods=1)
	returns = returns.drop('Close', 1)
	for i in range(1, len(stocks)):
		data = pandas_datareader.data.get_data_yahoo(stocks[i], start_date, end_date)[['Close']]
		returns['%s_daily_return'%stocks[i]] = data['Close'].pct_change(periods=1)
	returns = returns.dropna()
	return returns
  

returns = get_stocks_returns(stocks, start_date, end_date) 
returns = returns.as_matrix()
returns = returns.T
# plot the stocks return
plt.plot(returns.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
plt.show()

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_random_portfolios = 800
means, stds = np.column_stack([
    random_portfolio(returns) 
    for _ in xrange(n_random_portfolios)
])

plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

p_weights, p_returns, p_risks = optimal_portfolio(returns)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(p_risks, p_returns, 'y-o')
plt.show()

print p_weights
