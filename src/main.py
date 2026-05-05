import numpy as np
import matplotlib.pyplot as plt
import one_dim_methods as odm
from math import acos
from math import exp

func_counter = 0

def reset_counter():
    global func_counter
    func_counter = 0


def quadratic_function(x, y):
    global func_counter
    func_counter += 1
    return 100 * (y-x) ** 2 + (1-x) ** 2


def rosenbrock_function(x, y):
    global func_counter
    func_counter += 1
    return 100 * (y - x ** 2) ** 2 + (1-x) ** 2


def personal_function(x, y):
    global func_counter
    func_counter += 1
    return -(np.exp(-((x-3) / 1 ) ** 2 - ((y-1) / 3) ** 2) + 2 * np.exp(-((x-2) / 2) ** 2 - ((y-2) / 1) ** 2))


def gradient(x, y, function):
    grad = np.zeros(2)
    grad[0] = odm.partial_derivative_x(x, y, function)
    grad[1] = odm.partial_derivative_y(x, y, function)
    return grad



def find_lamda(x, y, function, one_dim_method):
    lamda_interval = odm.lamda_interval_search(x, y, function)
    lamda = one_dim_method(x, y, function,
                           a=lamda_interval[0], b=lamda_interval[1])
    return lamda


def gradient_decent(function, one_dim_method=odm.golden_section_method, eps=1e-5):
    coords_prev = np.zeros(2)
    coords_next = np.zeros(2)
    delta_func = 1.0
    
    #coords_prev[0] = input("Enter x0 ")
    #coords_prev[1] = input("Enter y0 ")

    coords_prev[0] = 4.
    coords_prev[1] = -2.

    x_values = np.array(coords_prev[0])
    y_values = np.array(coords_prev[1])
    
    i = 0
    print("TABLE 2 FOR GRADIENT DECENT")
    print(f'iter num | '
          f'{"coords":^32} | '
          f'{"func_value":^15} | '
          f'{"direction":^32} | '
          f'{"lamda":^15} | '
          f'{"delta coords":^32} | '
          f'{"delta func":^15} | '
          f'{"angle":^15} | '
          f'{"gradient":^32} | ')
    
    while abs(delta_func) > eps:
        direction = -gradient(coords_prev[0], coords_prev[1], function)
        lamda = find_lamda(coords_prev[0], coords_prev[1], 
                           function, one_dim_method)
        
        coords_next = coords_prev + lamda * direction

        func_value_next = function(coords_next[0], coords_next[1])
        func_value_prev = function(coords_prev[0], coords_prev[1])
        
        delta_coords = coords_next - coords_prev
        delta_func = func_value_next - func_value_prev
        
        grad_next = gradient(coords_next[0], coords_next[1], function)
        
        angle = acos(np.dot(coords_next, direction) /
                     np.linalg.norm(coords_next) / np.linalg.norm(direction))
        
        x_values = np.append(x_values, [coords_next[0]])
        y_values = np.append(y_values, [coords_next[1]])
        
        i += 1
        print(f'{i:^8} | '
              f'{coords_next[0]:^15.8e}  {coords_next[1]:^15.8e} | '
              f'{func_value_next:^15.8e} | '
              f'{direction[0]:^15.8e}  {direction[1]:^15.8e} | '
              f'{lamda:^15.8e} | '
              f'{delta_coords[0]:^15.8e}  {delta_coords[1]:^15.8e} | '
              f'{delta_func:^15.8e} | '
              f'{angle:^15.8e} | '
              f'{grad_next[0]:^15.8e}  {grad_next[1]:^15.8e} | ')
        
        coords_prev = coords_next.copy()
    """
    print("\nINFO FOR TABLE1\n")
    print(f'eps   | iter_count | func_counter | {"coords":^30} | func value  ')
    print(f'{eps} | {i:^10} | {func_counter:^12} | {coords_next[0]:.8e}  '
          f'{coords_next[1]:.8e} | {func_value_next:.8e}')
    """
    
    reset_counter()
    return x_values, y_values


def find_Hesse(delta_coords, delta_grad, Hesse):
    I = np.eye(2)
    denom  = np.dot(delta_grad, delta_coords)
    if(denom <= 1e-10):
        return I
    coeff = 1.0 / denom
    A1 = I - coeff * np.outer(delta_coords, delta_grad)
    A2 = I - coeff * np.outer(delta_grad, delta_coords)
    Hesse = np.dot(A1, np.dot(Hesse, A2)) + coeff * np.outer(delta_coords,delta_coords)
    return Hesse


def broyden_method(function, one_dim_method=odm.golden_section_method, eps=1e-5):
    coords_prev = np.zeros(2)
    coords_next = np.zeros(2)
    delta_func = 1.0
    
    I = np.eye(2)
    Hesse = I
    
    #coords_prev[0] = input("Enter x0 ")
    #coords_prev[1] = input("Enter y0 ")

    coords_prev[0] = 4.
    coords_prev[1] = -2.

    x_values = np.array(coords_prev[0])
    y_values = np.array(coords_prev[1])

    i = 0

    print("TABLE 2 FOR BROYDEN METHOD")
    print(f'{"Hesse":^25} | '
          f'iter num | '
          f'{"coords":^32} | '
          f'{"func_value":^15} | '
          f'{"direction":^32} | '
          f'{"lamda":^15} | '
          f'{"delta coords":^32} | '
          f'{"delta func":^15} | '
          f'{"angle":^15} | '
          f'{"gradient":^32} | ')
    
    while abs(delta_func) > eps:
        grad_prev = gradient(coords_prev[0], coords_prev[1], function)
        direction = -Hesse.dot(grad_prev)  
        lamda = find_lamda(coords_prev[0], coords_prev[1], 
                           function, one_dim_method)
        
        coords_next = coords_prev + lamda * direction
        grad_next = gradient(coords_next[0], coords_next[1], function)
        
        delta_coords = coords_next - coords_prev
        delta_grad = grad_next - grad_prev

        Hesse = find_Hesse(delta_coords, delta_grad, Hesse)
        
        func_value_next = function(coords_next[0], coords_next[1])
        func_value_prev = function(coords_prev[0], coords_prev[1])
        
        delta_func = func_value_next - func_value_prev

        angle = acos(np.dot(coords_next, direction) /
                     np.linalg.norm(coords_next) / np.linalg.norm(direction))
        
        x_values = np.append(x_values, [coords_next[0]])
        y_values = np.append(y_values, [coords_next[1]])
        
        i += 1
        print(f'{Hesse:} | '
              f'{i:^8} | '
              f'{coords_next[0]:^15.8e}  {coords_next[1]:^15.8e} | '
              f'{func_value_next:^15.8e} | '
              f'{direction[0]:^15.8e}  {direction[1]:^15.8e} | '
              f'{lamda:^15.8e} | '
              f'{delta_coords[0]:^15.8e}  {delta_coords[1]:^15.8e} | '
              f'{delta_func:^15.8e} | '
              f'{angle:^15.8e} | '
              f'{grad_next[0]:^15.8e}  {grad_next[1]:^15.8e} | ')

        
        coords_prev = coords_next.copy()
    """
    print("\nINFO FOR TABLE1\n")
    print(f'eps   | iter_count | func_counter | {"coords":^30} | func value  ')
    print(f'{eps} | {i:^10} | {func_counter:^12} | {coords_next[0]:.8e}  '
          f'{coords_next[1]:.8e} | {func_value_next:.8e}')
    """
    
    reset_counter()
    return x_values, y_values


def plot(function):
    x = np.linspace(-10, 10, 100)
    y = x
    
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = function(xgrid, ygrid)
    
    plt.figure(figsize=(5, 5))

    plt.contourf(xgrid, ygrid, zgrid, np.linspace(0, 2000, 2000))
    #plt.contourf(xgrid, ygrid, zgrid, np.linspace(-3, 2, 50))

    plt.colorbar()

    x_values_broyden, y_values_broyden = broyden_method(function)
    x_values_grad, y_values_grad = gradient_decent(function)
    
    plt.plot(x_values_broyden, y_values_broyden, 'x--r', label='broyden')
    plt.plot(x_values_grad, y_values_grad, 'x--g', label='gradient decent')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #plot(quadratic_function)
    plot(rosenbrock_function)
    # 0 2000 2000
    #plot(personal_function)
    # -3 2 50

    # Подозрения на некорректный сброс матрицы в методе Бройдена
    # Квадратичная функция сходится на 2ом шаге
    # Функция Розенброка погнао работает на ноуте. Разобраться почему
    # Функция варианта должна находить направление в любой ситуации. Даже в точке за пределами