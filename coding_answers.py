import numpy as np
#import matplotlib.pyplot as plt

#################### value functions for f_2 and f_3 ###############################

def f_2(x):
    return np.cos((x[0]+2)**2 + (x[1]-1)**2) + 4*x[0]**2 - 4*x[0]*(x[1]-1)**2

def f_3(x):
    return 1 - (np.exp(-x[0]**2-(x[1]-1)**2) + np.exp(-4*((x[0]+2)**2-(x[0]+2)*(x[1]-1)+(x[1]-1)**2))) - 0.1*np.log(x[0]**2/100+x[1]**2/100+(1/100)**2)



#################### Exercise 1c) ###############################
def grad_f1(x):
    return np.array([6*x[0]-4*x[1]+2, 6*x[1]-4*x[0]])

#gradient function df_2/dx- takes numpy (2, ) array inputs and return numpy (2, ) outputs
def grad_f2(x):
    d_x1 = -np.sin((x[0]+2)**2 + (x[1]-1)**2) * (2*x[0]+4) + 8*x[0] - 4*(x[1]-1)
    d_x2 = -np.sin((x[0]+2)**2 + (x[1]-1)**2) * (2*x[1]-1) - 4*x[0] + 8*(x[1]-1)
    return np.array([d_x1, d_x2])

#gradient function df_3/dx- takes numpy (2, ) array inputs and return numpy (2, ) outputs
def grad_f3(x):
    d_x1 = 2*x[0]*np.exp(-x[0]**2-(x[1]-1)**2) + (8*(x[0]+2)-4*(x[1]-1))*np.exp(-4*((x[0]+2)**2-(x[0]+2)*(x[1]-1)+(x[1]-1)**2)) - x[0]/(5*(x[0]**2+x[1]**2+0.01))
    d_x2 = 2*(x[1]-1)*np.exp(-x[0]**2-(x[1]-1)**2) + (8*(x[1]-1)-4*(x[0]+2))*np.exp(-4*((x[0]+2)**2-(x[0]+2)*(x[1]-1)+(x[1]-1)**2)) - x[1]/(5*(x[0]**2+x[1]**2+1/100))
    return np.array([d_x1, d_x2])



#################### Exercise 1d) ###############################
#gradient descent algorithm for f_2 - takes as input: the start point as 1x2 array and the number of iterations
# output: array of gradient points (size iterationsx2), calculated gradient value

def gradient_descent(start_point, iterations, gradient_function, gamma, t = False):
    gradient_steps = np.zeros([iterations+1,2])
    gradient_steps[0] = start_point
    if not t:
        for i in range(gradient_steps.shape[0]-1):
            gradient_steps[i+1] = gradient_steps[i] - gamma*gradient_function(gradient_steps[i])
    else:
        for i in range(gradient_steps.shape[0]-1):
            gradient_steps[i+1] = gradient_steps[i] - gamma/(1+i)*gradient_function(gradient_steps[i])
    return gradient_steps, gradient_function(gradient_steps[-1])

############################# Execution ############################

if __name__ == '__main__':
    start_point = np.array([0.3,0])
    iterations = 50

    #I use a gamma of 1/t per iteration
    points_f2, gradient_value_f2 = gradient_descent(start_point, iterations, grad_f2, 1, True)
    print('found local minimum for function 2 at estimated point {}, with gradient values of {}, and function value f2({}) = {}'.format(points_f2[-1], gradient_value_f2, points_f2[-1], f_2(points_f2[-1])))

    #I use a gamma of 1/t per iteration
    points_f3, gradient_value_f3 = gradient_descent(start_point, iterations, grad_f3, 1, True)
    print('found local minimum for function 3 at estimated point {}, with gradient values of {}, and function value f2({}) = {}'.format(points_f3[-1], gradient_value_f3, points_f3[-1], f_3(points_f2[-1])))




############################## plot functions ############################

def draw_contour_plot(function, start_point, iterations, gradient_function, gamma, t = False):
    x_min = min(gradient_descent(start_point, iterations, gradient_function,gamma,t)[0][:,0]) - 0.1
    y_min = min(gradient_descent(start_point, iterations, gradient_function,gamma,t)[0][:,1]) - 0.1
    x_max = max(gradient_descent(start_point, iterations, gradient_function,gamma,t)[0][:,0]) + 0.1
    y_max = max(gradient_descent(start_point, iterations, gradient_function,gamma,t)[0][:,1]) + 0.1
    x1 = np.linspace(x_min - abs(0.1*x_min) - 0.1, x_max + abs(0.1*x_max), 200)
    x2 = np.linspace(y_min - abs(0.1*y_min), y_max + abs(0.1*y_max), 200)
    function_vals = np.zeros(shape=(x1.size, x2.size))

    for i, value1 in enumerate(x1):
        for j, value2 in enumerate(x1):
            x_temp = np.array([value1,value2])
            function_vals[i, j] = function(x_temp)

    fig = plt.contourf(x1, x2, function_vals)
    cbar = plt.colorbar(fig)

    gradient_points, gradient_value = gradient_descent(start_point, iterations, gradient_function,gamma,t)
    plt.plot(gradient_points[:,0],gradient_points[:,1], 'r-o', markersize = 3)
    plt.title("Contour Plot of Gradient Descent")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.show
    plt.savefig('grad.png')
    print('Minimum at: x_1={}, x_2={}, gradient_value={}'.format(gradient_points[-1][0],gradient_points[-1][1], gradient_value))

draw_contour_plot(f_2, [0.3,0], 50, grad_f2, 1, True)



def plot_step_sizes(start_point, iterations, gradient_function, gamma_limit, t = False):
    gamma = np.linspace(0.01, gamma_limit, 100)
    mae = np.zeros(100)
    for i in range(len(mae)):
        mae[i] = abs(gradient_descent(start_point, iterations, gradient_function, gamma[i])[1]).mean()
    plt.plot(gamma, mae)
    plt.xlabel("gamma")
    plt.ylabel("Mean absolute error")
    plt.savefig('gammas.png')

