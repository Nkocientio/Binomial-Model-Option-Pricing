import math
import random
import numpy as np
import pandas as pd
import streamlit as st
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

        self.stock_tree()

    def stock_tree(self):
        self.strikes = np.full(self.N+1,self.K)
        self.stock = np.zeros((self.N+1,self.N+1))
        self.S0 = max(self.S0 - np.sum(self.divs[1:,:]), 0.0)
        self.div_sum = np.zeros(self.N+1)
        
        u = self.u
        if self.style == 'Asian':
            if self.avg_style == 'Arithmetic':
                self.agg_stock = self.stock.copy()
            else:
                u = np.sqrt(u)
            scalar_s = self.S0*u**self.N
        
        for j in range(self.N+1):
            #add all future divs
            idx = np.argmax(self.divs[0,:] > self.dt*j)
            self.div_sum[j] = ( np.sum(self.divs[1,idx::]) if self.divs[0,idx] > self.dt*j else 0 ) * np.exp(self.r*self.dt * j)

            #powers of the asset 
            powers = np.arange(j ,-j-1,-2)
            asset_values = self.S0*u**powers
            
            asset_values += self.div_sum[j] 
            self.stock[0:j+1,j] = asset_values
            
            if self.style == 'Asian':
                if self.avg_what == 'Asset':
                    if self.avg_style == 'Arithmetic':
                        self.asian_stock(j,scalar_s)
                else:
                    self.strikes[j] = np.random.randint(0.95*self.K,1.05*self.K)
        self.strikes[0] = self.K
        
    def asian_stock(self,j,scalar_s):
        # Asset average, scalar_s is just used to prevent the overflow error
        # At node (i,j) there are num_paths ways to get there, also considering the length (j+1). (j for col, and for row)
        row_ref = j
        for i in range(j+1):
            stock = self.stock[i,j] / scalar_s
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
            
            self.stock[i,j] = scalar_s * self.agg_stock[i,j]/( (j+1)*num_paths )

    
    def build_tree(self):
        N = self.N
        self.can_exercise = self.exerciseOn(N,self.double_style)
        
        asset_values = self.stock[:,-1]
        terminal_values = np.maximum(self.epsilon*(asset_values-self.strikes[-1]),0.0)
        
        tree = np.empty((2*N+2,N+1),dtype=object)
        self.font_colors = np.full( (2*N+2,N+1) ,False, dtype=bool) # indicator: if the holder can exercise
        
        tree[0::2,N] = asset_values
        tree[1::2,N] = terminal_values

        self.font_colors[1::2,N] = True & (terminal_values > 0.0)
        
        for j in reversed(range(N)):
            asset_values = self.stock[0:j+1,j]
            
            row_1 = terminal_values[:j+1]
            row_2 = terminal_values[1:j+2]
            rows_2d = np.array([row_1, row_2])
            
            values = self.probs.dot(rows_2d)

            # lb: lowerBound, ub: upperBound
            lb = N-j
            ub = 2*N+2-lb
            if self.can_exercise[j]:
                intrinsic = np.maximum( self.epsilon*(asset_values-self.strikes[:j+1]), 0.0)
                terminal_values = np.maximum(intrinsic,values*self.df)

                exercise = np.full((2*j+2), False, dtype = bool)
                exercise[1::2] = (intrinsic == terminal_values) & (intrinsic > 0)
                self.font_colors[lb:ub,j] = exercise
            else:
                terminal_values = values*self.df
        
            col = np.zeros(2*j+2)
            col[0::2] = asset_values
            col[1::2] = terminal_values
        
            tree[lb:ub,j] = col
            
        self.price = terminal_values[0]
        self.tree  = tree
        
    def compound_option(self,option_two,K1,m):
        self.epsilon = -1 if option_two == 'Put' else 1
        self.option_two, self.K1, self.n1 = option_two, K1, m
        self.build_tree()

        N = self.N
        lb = N-m
        ub = 2*N+2-lb
        
        asset_values = self.tree[lb:ub,m][1::2]
        terminal_values = np.maximum(self.epsilon*(asset_values-K1),0.0)

        col = np.zeros(2*m+2)
        col[0::2] = asset_values
        col[1::2] = terminal_values
        
        self.tree[lb:ub,m] = col
        
        exercise = np.full((2*m+2), False, dtype = bool)
        exercise[1::2] = True & (terminal_values > 0)
        self.font_colors[lb:ub,m] = exercise
        
        for j in reversed(range(m)):
            lb, ub = N-j , 2*N+2-lb
            asset_values = self.tree[lb:ub,j][1::2]
            
            row_1 = terminal_values[:j+1]
            row_2 = terminal_values[1:j+2]
            array_2d = np.array([row_1, row_2])
            
            values = self.probs.dot(array_2d)
            
            if self.can_exercise[j]:
                intrinsic = np.maximum( self.epsilon*(asset_values-K1), 0.0)
                terminal_values = np.maximum(intrinsic,values*self.df)

                exercise = np.full((2*j+2), False, dtype = bool)
                exercise[1::2] = (intrinsic == terminal_values) & (intrinsic > 0)
                self.font_colors[lb:ub-1,j] = exercise
            else:
                terminal_values = values*self.df
        
            col = np.zeros(2*j+2)
            col[0::2] = asset_values
            col[1::2] = terminal_values
        
            self.tree[lb:ub-1,j] = col
            
        self.price = terminal_values[0]
        
    def exerciseOn(self,n,style):
        self.b_dates = []
        if style == 'American':
            return np.full(n, True, dtype=bool)
        elif style == 'Bermudan':
            self.exercise_dates = np.array(self.exercise_dates, dtype='datetime64[D]')
            arr = np.full(n+1, False, dtype=bool)
            today = np.datetime64(datetime.now().strftime('%Y-%m-%d'))
            for dt in self.exercise_dates:
                delta = (dt - today) 
                yrs = delta.astype('timedelta64[D]').astype(int) / 365
                node = int(yrs/self.dt )
                arr[min(node,n)] = True
                self.b_dates.append(f"{dt.astype(datetime).strftime('%d %b %Y')} ({node})")
            return arr
        else:
            return np.full(n, False, dtype=bool)

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
        n1 = self.N
        if self.style == 'Asian':
            d_style = self.double_style
            avg_what = self.avg_what
            avg_style = self.avg_style
        elif self.style == 'Compound':
            d_style = self.double_style
            option_two = self.option_two
            K1 = self.K1
            n1 = self.n1
        
        params= [self.style,self.option_type,self.tree,self.font_colors,self.N,self.u,self.probs[0],self.strikes,d_style,dividends,self.dt,option_two,K1,n1,avg_style,avg_what,self.S0,self.K,self.T,self.r,self.q,self.sigma,self.price,self.b_dates, self.div_type]
        wb = print_excel(*params)
        return get_workbook_as_bytes(wb)
        
