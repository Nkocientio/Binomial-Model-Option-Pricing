import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime,timedelta
from code_excel import print_excel,get_workbook_as_bytes

class Binomial_Model():
    def __init__(self,S0,K,T,r,q,sigma,style,double_style,option_type,N,divs,avg_what,avg_style):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.style = style
        self.double_style = double_style
        self.option_type = option_type
        self.N = N
        self.avg_what, self.avg_style = avg_what,avg_style
        self.divs = divs
        self.intermediate_values()

    def intermediate_values(self):
        self.dt = self.T/self.N
        self.epsilon = -1 if self.option_type == 'Put' else 1
        self.df = np.exp(-self.dt*self.r)
        
        self.u = np.exp( self.sigma * np.sqrt(self.dt) )
        u = self.u
        d =  1/self.u
        p = ( np.exp((self.r-self.q)*self.dt) - d) / (u -d)
        self.probs = np.array([p,1-p])

        # order the Cash divs by year and discount them to T0
        self.divs[1,:] = self.divs[1,:]*np.exp(-self.r*self.divs[0,:])
        sorted_indices = np.argsort(self.divs[0, :]) 
        self.divs = self.divs[:, sorted_indices]

        self.calculate_stock_values()

    def calculate_stock_values(self):
        self.strikes = np.full(self.N+1,self.K)
        self.asset_values = np.zeros((self.N+1,self.N+1))
        self.S0 = max(self.S0 - np.sum(self.divs[1:,:]), 0.0)
        self.div_sum = np.zeros(self.N+1)
        
        u = self.u
        if self.style == 'Asian':
            if self.avg_style == 'Arithmetic':
                self.agg_stock = self.asset_values.copy()
            else:
                u = np.sqrt(u)
            scalar_s = self.S0*u**self.N
        
        for j in range(self.N+1):
            #add all future divs
            idx = np.argmax(self.divs[0,:] > self.dt*j)
            self.div_sum[j] = ( np.sum(self.divs[1,idx::]) if self.divs[0,idx] > self.dt*j else 0 ) * np.exp(self.r*self.dt * j)

            #powers of the asset 
            powers = np.arange(j ,-j-1,-2)
            asset_col = self.S0*u**powers
            
            asset_col += self.div_sum[j] 
            self.asset_values[0:j+1,j] = asset_col
            
            if self.style == 'Asian':
                if self.avg_what == 'Asset':
                    if self.avg_style == 'Arithmetic':
                        self.asian_stock(j,scalar_s)
                else:
                    self.strikes[j] = np.random.randint(0.95*self.K,1.05*self.K)
        self.strikes[0] = self.K
        
    def asian_stock(self,j,scalar_s):
        # Asset average, scalar_s is just used to prevent the overflow error
        # At node (i,j) there are num_paths ways to get there, also considering the length (j+1). (j for col, and i for row)
        row_ref = j
        for i in range(j+1):
            stock = self.asset_values[i,j] / scalar_s
            if i == 0:
                num_paths = 1
                self.agg_stock[i,j] = self.agg_stock[ i,max(0,j-1)] + stock
            elif i == j:
                num_paths = 1
                self.agg_stock[i,j] = self.agg_stock[ i-1,j-1 ] + stock
            else:
                row_ref -= 1
                num_paths = math.comb(j,row_ref)
                self.agg_stock[i,j] = self.agg_stock[ i-1,j-1 ] + self.agg_stock[ i,j-1 ] + num_paths*stock
            
            self.asset_values[i,j] = scalar_s * self.agg_stock[i,j]/( (j+1)*num_paths )

    
    def calculate_option_values(self):
        N = self.N
        self.can_exercise = self.exerciseOn(N,self.double_style)
       
        calculated_values = np.maximum(self.epsilon*( self.asset_values[:,-1] - self.strikes[-1] ),0.0)
        
        self.option_values = np.zeros((N+1,N+1))
        self.option_exercised = np.full( (N+1,N+1) ,False, dtype=bool) # indicator: if the holder can exercise

        self.option_values[:,-1] = calculated_values
        self.option_exercised[:,-1] = True & (calculated_values > 0.0)
        
        for j in reversed(range(N)):
            row_1 = calculated_values[:j+1]
            row_2 = calculated_values[1:j+2]
            rows_2d = np.array([row_1, row_2])
            
            values = self.probs.dot(rows_2d)

            if self.can_exercise[j]:
                intrinsic = np.maximum( self.epsilon*( self.asset_values[:j+1,j] - self.strikes[:j+1] ), 0.0)
                calculated_values = np.maximum(intrinsic,values*self.df)
                self.option_exercised[:j+1,j] = (intrinsic == calculated_values) & (intrinsic > 0)
            else:
                calculated_values = values*self.df
        
            self.option_values[:j+1,j] = calculated_values
            
        self.price = calculated_values[0]
        
    def compound_option(self,option_two,K1,T1):
        self.epsilon = -1 if option_two == 'Put' else 1
        self.option_two, self.K1, self.T1 = option_two, K1, T1
        
        self.calculate_option_values()

        m = max(1, int(self.T1/self.dt))
        self.asset_values[:m+1,m] = self.option_values[:m+1,m]
        calculated_values = np.maximum(self.epsilon*( self.asset_values[:m+1,m] - self.K1 ), 0.0)
        self.option_values[:m+1,m] = calculated_values
        
        self.option_exercised[:m+1,m] = True & (calculated_values > 0)
        
        for j in reversed(range(m)):
            self.asset_values[:j+1,j] = self.option_values[:j+1,j]
            
            row_1 = calculated_values[:j+1]
            row_2 = calculated_values[1:j+2]
            array_2d = np.array([row_1, row_2])
            
            values = self.probs.dot(array_2d)
            
            if self.can_exercise[j]:
                intrinsic = np.maximum( self.epsilon*( self.asset_values[:j+1,j] - self.K1 ), 0.0)
                calculated_values = np.maximum(intrinsic, values*self.df)

                self.option_exercised[:j+1,j] = (intrinsic == calculated_values) & (intrinsic > 0)
            else:
                calculated_values = values*self.df
        
            self.option_values[:j+1,j] = calculated_values
            
        self.price = calculated_values[0]
        
    def exerciseOn(self,n,style):
        self.b_dates = []
        if style == 'American':
            return np.full(n, True, dtype=bool)
        elif style == 'Bermudan':
            arr = np.full(n+1, False, dtype=bool)
            today = np.datetime64(datetime.now().strftime('%Y-%m-%d'))
            self.exercise_dates = np.array(self.exercise_dates, dtype='datetime64[D]')
            for date in self.exercise_dates:
                days = (date - today) 
                yrs = days.astype('timedelta64[D]').astype(int) / 365
                node = int(yrs/self.dt)
                arr[min(node,n)] = True
                self.b_dates.append(f"{date.astype(datetime).strftime('%d %b %Y')} ({node})")
            return arr
        else:
            return np.full(n, False, dtype=bool)  #Eur style

    def black_scholes(self):
        discount = np.exp(-self.r*self.T)
        forward_price = self.S0*np.exp((self.r-self.q)*self.T) 
        vol_sqrt_T = self.sigma*np.sqrt(self.T)
        
        d1 = np.log(forward_price/self.K) / vol_sqrt_T + 0.5*vol_sqrt_T
        d2 = d1 - vol_sqrt_T
        
        price = self.epsilon * discount * (forward_price*norm.cdf(self.epsilon*d1) - self.K*norm.cdf(self.epsilon*d2))
        return price

    def excel_values(self):
        dividends = self.div_sum
        
        avg_what, avg_style = '',''
        d_style = ''
        option_two = 'Call'
        K1 = self.K
        T1 = self.T
        if self.style == 'Asian':
            d_style = self.double_style
            avg_what = self.avg_what
            avg_style = self.avg_style
        elif self.style == 'Compound':
            d_style = self.double_style
            option_two = self.option_two
            K1 = self.K1
            T1 = self.T1
        
        params= [self.style,self.option_type,self.asset_values,self.option_values,self.option_exercised,self.N,self.u,self.probs[0],self.strikes,d_style,dividends,self.dt,option_two,K1,T1,avg_style,avg_what,self.S0,self.K,self.T,self.r,self.q,self.sigma,self.price,self.b_dates, self.div_type]
        wb = print_excel(*params)
        return get_workbook_as_bytes(wb)
        
