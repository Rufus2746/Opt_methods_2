import numpy as np
from math import sqrt
from math import log

def partial_derivative_x(x, y, function, delta_x=1e-9):
    return (function(x + delta_x, y) - function(x, y)) / delta_x

def partial_derivative_y(x, y, function, delta_y=1e-9):
    return (function(x, y + delta_y) - function(x, y)) / delta_y

def fibonacci_num(n):
    return (((1 + sqrt(5)) / 2) ** n - ((1 - sqrt(5)) / 2) ** n) / sqrt(5)

def find_n_for_fibonacci(*, a=-2, b=20, eps):
    t1 = log((1 + sqrt(5)) / 2)
    t2 = log(sqrt(5) * (b - a) * 4 / eps / ((1+sqrt(5)) ** 2))
    n = int(t2 / t1 + 1)
    return n

def lamda_interval_search(x, y, function):
    sigma = 0.0001
    lmbda = 0.0
    
    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)
    
    def f_line(l):
        return function(x - l * x_der, y - l * y_der)

    f0 = f_line(lmbda)
    f_plus = f_line(lmbda + sigma)
    
    if f0 > f_plus:
        h = sigma
        lmbda_next = lmbda + sigma
    else:
        f_minus = f_line(lmbda - sigma)
        if f_minus < f0:
            h = -sigma
            lmbda_next = lmbda - sigma
        else:
            return np.array([lmbda - sigma, lmbda + sigma])

    lmbda_prev = lmbda
    f_prev = f0
    f_curr = f_line(lmbda_next)
    
    while f_curr < f_prev:
        h *= 2
        lmbda_prev = lmbda_next
        f_prev = f_curr
        
        lmbda_next = lmbda_prev + h
        f_curr = f_line(lmbda_next)
    
    points = sorted([lmbda_prev - h/2, lmbda_next]) 
    return np.array(points)

def lamda_interval_search_old(x, y, function):
    sigma = 0.0001
    lamda_prev = 0.
    k = 1
    
    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)
    
    x_prev = x - lamda_prev * x_der
    y_prev = y - lamda_prev * y_der
    
    x_next = x - (lamda_prev + sigma) * x_der
    y_next = y - (lamda_prev + sigma) * y_der
    
    if function(x_prev, y_prev) > function(x_next, y_next):
        lamda_next = lamda_prev + sigma
        h = sigma
    else:
        lamda_next = lamda_prev - sigma
        h = -sigma

    flag = True
    
    while flag:
        h *= 2
        lamda_next += h
        
        x_prev, y_prev = x_next, y_next
        
        x_next = x - lamda_next * x_der
        y_next = y - lamda_next * y_der

        k += 1
        if function(x_prev, y_prev) > function(x_next, y_next):
            lamda_prev = lamda_next
        else:
            flag = False

    if lamda_prev < lamda_next:
        return np.array([lamda_prev-h/2, lamda_next])
    else:
        return np.array([lamda_next, lamda_prev+h/2]) 
      
def dichotomy(x, y, function, a=-2, b=20, eps=1e-7):
    a1 = a
    b1 = b
    
    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)
    
    i = 1
    while b1 - a1 > eps:
        delta = eps/10

        lamda1 = (a1 + b1 - delta) / 2
        lamda2 = (a1 + b1 + delta) / 2
        
        x1 = x - lamda1 * x_der
        y1 = y - lamda1 * y_der
        
        x2 = x - lamda2 * x_der
        y2 = y - lamda2 * y_der

        if function(x1, y1) < function(x2, y2):
            b1 = lamda2
        else:
            a1 = lamda1
            
        i += 1
    return (a1 + b1) / 2

def parabola_method(x, y, function, a=-2, b=20, eps=1e-7,  h=0.5):
    lamda_middle = (a + b) / 2
    lamda_min = - lamda_middle
    
    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)

    while abs(lamda_min - lamda_middle) > eps:
        lamda_left = lamda_middle - h
        lamda_right = lamda_middle + h
        
        x_left = x - lamda_left * x_der
        y_left = y - lamda_left * y_der
        
        x_right = x - lamda_right * x_der
        y_right = y - lamda_right * y_der
        
        x_middle = x - lamda_middle * x_der
        y_middle = y - lamda_middle * y_der
        
        lamda_min = 0.5 * ((function(x_left, y_left) * 
                            (lamda_right + lamda_middle) - 2 * 
                            function(x_middle, y_middle) * 
                            (lamda_right + lamda_left) + 
                            function(x_right, y_right) * 
                            (lamda_middle + lamda_left)) /
                           (function(x_left, y_left) - 2 * 
                            function(x_middle, y_middle) + 
                            function(x_right, y_right)))
        
        h = abs(lamda_min - lamda_middle)
        lamda_middle = lamda_min
    return lamda_min

def golden_section_method(x, y, function, a=-2, b=20, eps=1e-7):
    a1 = a
    b1 = b

    coeff_1 = (3 - sqrt(5)) / 2
    coeff_2 = (sqrt(5) - 1) / 2

    lamda1 = a + coeff_1 * (b - a)
    lamda2 = a + coeff_2 * (b - a)

    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)

    x1 = x - lamda1 * x_der
    y1 = y - lamda1 * y_der
    
    x2 = x - lamda2 * x_der
    y2 = y - lamda2 * y_der
    
    func_value_1 = function(x1, y1)
    func_value_2 = function(x2, y2)
    
    i = 0
    while b1 - a1 > eps:
        a2 = a1
        b2 = b1
        
        if func_value_1 < func_value_2:
            b2 = lamda2
            lamda2 = lamda1
            x2 = x1
            y2 = y1
            func_value_2 = func_value_1
            lamda1 = a2 + coeff_1 * (b2 - a2)
            x1 = x - lamda1 * x_der
            y1 = y - lamda1 * y_der
            func_value_1 = function(x1, y1)
        else:
            a2 = lamda1
            lamda1 = lamda2
            x1 = x2
            y1 = y2
            func_value_1 = func_value_2
            lamda2 = a2 + coeff_2 * (b2 - a2)
            x2 = x - lamda2 * x_der
            y2 = y - lamda2 * y_der
            func_value_2 = function(x2, y2)
            
        b1 = b2
        a1 = a2
        i += 1
    return (a1 + b1) / 2

def fibonacci_method(x, y, function, a=-2, b=20, eps=1e-7):
    initial_length = b - a
    a1 = a
    b1 = b

    n = find_n_for_fibonacci(eps=1e-7)

    Fn2 = fibonacci_num(n + 2)

    lamda1 = a + fibonacci_num(n) / Fn2 * initial_length
    lamda2 = a + fibonacci_num(n + 1) / Fn2 * initial_length
    
    x_der = partial_derivative_x(x, y, function)
    y_der = partial_derivative_y(x, y, function)

    x1 = x - lamda1 * x_der
    y1 = y - lamda1 * y_der
    
    x2 = x - lamda2 * x_der
    y2 = y - lamda2 * y_der
    
    func_value_1 = function(x1, y1)
    func_value_2 = function(x2, y2)

    i = 1
    while i <= n:
        b2 = b1
        a2 = a1
        
        if func_value_1 < func_value_2:
            b2 = lamda2
            lamda2 = lamda1
            x2 = x1
            y2 = y1
            func_value_2 = func_value_1
            lamda1 = a2 + fibonacci_num(n - i) / Fn2 * initial_length
            x1 = x - lamda1 * x_der
            y1 = y - lamda1 * y_der
            func_value_1 = function(x1, y1)
        else:
            a2 = lamda1
            lamda1 = lamda2
            func_value_1 = func_value_2
            lamda2 = a2 + fibonacci_num(n - i + 1) / Fn2 * initial_length
            x2 = x - lamda2 * x_der
            y2 = y - lamda2 * y_der
            func_value_2 = function(x2, y2)

        b1 = b2
        a1 = a2
        i += 1
    return (a1 + b1) / 2
