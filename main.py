import matplotlib.pyplot as plt
import math
import random
import time
from tqdm import tqdm


class Object:
    """Represents an object with geographical position and likelihood information."""
    
    def __init__(self, u, v, x, y, logL, logWt):
        """
        Initialize Object attributes.

        Args:
            u (float): Uniform prior controlling parameter for x.
            v (float): Uniform prior controlling parameter for y.
            x (float): Geographical easterly position of the lighthouse.
            y (float): Geographical northerly position of the lighthouse.
            logL (float): Log-likelihood, natural logarithm of the probability of data given position.
            logWt (float): Logarithm of the weight, used for calculating evidence.
        """
        self.u = u
        self.v = v
        self.x = x
        self.y = y
        self.logL = logL
        self.logWt = logWt
import math

def import_txt_file_first_column_as_numbers(file_path):
    """
    Reads the first column of a text file as numbers.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: The numbers from the first column.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        first_column_data = [float(line.split()[0]) for line in lines]
    return first_column_data

def logLhood(x, y, arrival_positions):
    # x: easterly position
    # y: northerly position
    # data: a list of arrival positions of data points.
    logL = 0  # logLikelihood accumulator
    for k in range(len(arrival_positions)):
        logL += math.log((y / math.pi) / ((arrival_positions[k] - x) * (arrival_positions[k] - x) + y * y))
    return logL

def Prior(obj, arrival_position):
    # Set Object attributes according to prior distributions
    obj.u = random.uniform(0, 1)  # uniform in (0,1)
    obj.v = random.uniform(0, 1)  # uniform in (0,1)
    obj.x = 4.0 * obj.u - 2.0  # map to x
    obj.y = 4.0 * obj.v  # map to y
    obj.logL = logLhood(obj.x, obj.y, arrival_position)  # Calculate log likelihood using the provided function

def Explore(obj, logLstar, arrival_position):
    # Evolve object within likelihood constraint
    # obj: Object being evolved
    # logLstar: Likelihood constraint L > Lstar
    step = 0.1  # Initial guess suitable step-size in (0,1)
    m = 20  # MCMC counter (pre-judged # steps)
    accept = 0  # # MCMC acceptances
    reject = 0  # # MCMC rejections
    # print("logl:", obj.logL)
    # print("logLstar:", logLstar)
    for _ in range(m):
        Try = Object(0, 0, 0, 0, 0, 0)
        # Trial object
        Try.u = obj.u + step * (2.0 * random.random() - 1.0)  # |move| < step
        Try.v = obj.v + step * (2.0 * random.random() - 1.0)  # |move| < step
        Try.u -= math.floor(Try.u)  # wraparound to stay within (0,1)
        Try.v -= math.floor(Try.v)  # wraparound to stay within (0,1)
        Try.x = 4.0 * Try.u - 2.0  # map to x
        Try.y = 4.0 * Try.v  # map to y
        Try.logL = logLhood(Try.x, Try.y, arrival_position)  # trial likelihood value
        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            obj = Try
            # print("accept_try_logl:", Try.logL)
            # print("accept_obj_logl:", obj.logL)
            accept += 1
        else:
            reject += 1
        # Refine step-size to let acceptance ratio converge around 50%
        if accept > reject:
            step *= math.exp(1.0 / accept)
        if accept < reject:
            step /= math.exp(1.0 / reject)
    # print("explorelogl:", obj.logL)
    return obj

def Results(samples, nest, logZ):
    """
    Calculates the mean and standard deviation of the x and y attributes of the Object instances defining the posterior distribution.

    Parameters:
    - samples: list of Object instances representing posterior samples
    - nest: number of samples in the posterior distribution
    - logZ: evidence, which is the logarithm of the total weight of all samples

    Returns:
    None (prints the mean and standard deviation of x and y attributes)
    """

    x_mean = 0.0  # Mean of x
    xx_mean = 0.0  # Second moment of x
    y_mean = 0.0  # Mean of y
    yy_mean = 0.0  # Second moment of y
    
    for i in range(nest):
        w = math.exp(samples[i].logWt - logZ)
        x_mean += w * samples[i].x
        xx_mean += w * samples[i].x * samples[i].x
        y_mean += w * samples[i].y
        yy_mean += w * samples[i].y * samples[i].y
    
    x_stddev = math.sqrt(xx_mean - x_mean * x_mean)  # Standard deviation of x
    y_stddev = math.sqrt(yy_mean - y_mean * y_mean)  # Standard deviation of y
    
    # Print the mean and standard deviation of x and y
    print("mean(alpha) = {:.6f}, stddev(alpha) = {:.6f}".format(x_mean, x_stddev))
    print("mean(beta) = {:.6f}, stddev(beta) = {:.6f}".format(y_mean, y_stddev))


def PLUS(x, y):
    if x > y:
        return x + math.log(1 + math.exp(y - x))
    else:
        return y + math.log(1 + math.exp(x - y))
    
def plot_marginal_posterior(samples, nest, logZ):
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the marginal posterior distribution of y
    axs[0].hist([sample.y for sample in samples[:nest]], bins=100, weights=[math.exp(sample.logWt - logZ) for sample in samples[:nest]], color='salmon', edgecolor='black', alpha=0.7)
    axs[0].set_xlabel('$\\beta$')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Marginalised Posterior Distribution of $\\beta$')

    # Plot the marginal posterior distribution of x
    axs[1].hist([sample.x for sample in samples[:nest]], bins=100, weights=[math.exp(sample.logWt - logZ) for sample in samples[:nest]], color='skyblue', edgecolor='black', alpha=0.7)
    axs[1].set_xlabel('$\\alpha$')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Marginalised Posterior Distribution of $\\alpha$')

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_2d_histogram(samples, nest, logZ):
    # Extract x and y values from samples
    x_values = [sample.x for sample in samples[:nest]]
    y_values = [sample.y for sample in samples[:nest]]
    
    # Create a 2D histogram with contour lines
    plt.hist2d(x_values, y_values, bins=100, weights=[math.exp(sample.logWt - logZ) for sample in samples[:nest]], cmap='viridis')
    plt.colorbar(label='Frequency', aspect=20)

    # Add labels and title
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\beta$')
    plt.title('2D Histogram showing the joint posterior on $\\alpha$ and $\\beta$')

    # Show the plot
    plt.show()

def main():
    # Set the number of iterations
    total_iterations = 100000
    # Create a progress bar using tqdm
    progress_bar = tqdm(total=total_iterations, desc="Running")
    # Start timing
    start_time = time.time()

    # Main function's code logic
    n = 10000  # the number of Objects(live points)
    MAX = 100000  # the number of itration
    Obj = [Object(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n)]  # Collection of n objects
    Samples = [Object(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(MAX)]  # Objects stored for posterior results
    logwidth = 0.0  # Width in prior mass
    logLstar = 0.0  # Likelihood constraint
    H = 0.0  # Information, initially 0
    logZ = -1e300  # ln(Evidence Z), initially 0
    logZ_evolution = []  # Store the evolution of evidence Z
    # logZ = float('-inf')  # ln(Evidence Z), initially 0
    logZnew = 0.0  # Updated logZ
    i = 0  # Object counter
    copy = 0  # Duplicated object
    worst = 0  # Worst object
    nest = 0  # Nested sampling iteration count
    file_path = 'lighthouse_flash_data.txt'  # Replace with the path to your text file
    arrival_position = import_txt_file_first_column_as_numbers(file_path)
    # Set prior objects
    for i in range(n):
        Prior(Obj[i], arrival_position)
    # Outermost interval of prior mass
    logwidth = math.log(1.0 - math.exp(-1.0 / n))
    # Nested Sampling Loop
    for nest in range(MAX):
        # Worst object in collection, with Weight = width * Likelihood
        worst = 0
        for i in range(1, n):
            if Obj[i].logL < Obj[worst].logL:
                worst = i
        Obj[worst].logWt = logwidth + Obj[worst].logL
        # Update Evidence Z and Information H
        logZnew = PLUS(logZ, Obj[worst].logWt)
        H = math.exp(Obj[worst].logWt - logZnew) * Obj[worst].logL + math.exp(logZ - logZnew) * (H + logZ) - logZnew
        logZ = logZnew
        # Store the current value of evidence Z
        logZ_evolution.append(logZ)
        # Posterior Samples 
        Samples[nest] = Obj[worst]  # Uncomment if you want to store posterior samples
        # Kill worst object in favour of copy of different survivor
        copy = int(n * random.random()) % n  # force 0 <= copy < n
        while copy == worst and n > 1:  # don’t kill if n is only 1
            copy = int(n * random.random()) % n
        logLstar = Obj[worst].logL  # new likelihood constraint
        Obj[worst] = Obj[copy]  # overwrite worst object
        # Evolve copied object within constraint
        Obj[worst] = Explore(Obj[worst], logLstar, arrival_position)
        # Shrink interval
        logwidth -= 1.0 / n
        # Update progress bar
        progress_bar.update(1)
    # Exit with evidence Z, information H, and optional posterior Samples
    # Print the number of iterations
    print("# iterates =", nest+1)
    # Print evidence Z and information H
    print("Evidence: ln(Z) =", logZ, "+-", math.sqrt(H/n))
    # print("Information: H =", H, "nats =", H/math.log(2), "bits")
    # calculate and print posterior results
    Results(Samples, nest, logZ)  # optional
    plot_marginal_posterior(Samples, nest, logZ)
    plot_2d_histogram(Samples, nest, logZ)

    # Plot the evolution of evidence Z
    plt.plot(range(1, len(logZ_evolution)+1), logZ_evolution)
    plt.xlabel('Iteration')
    plt.ylabel('Evidence ln(Z)')
    plt.title('Evolution of Evidence Z')
    plt.show()

    # End timing
    end_time = time.time()
    # Close the progress bar
    progress_bar.close()
    # Calculate the execution time
    execution_time = end_time - start_time
    # Print a message to inform the programmer
    print("The code ran smoothly and took {:.2f} seconds in total.".format(execution_time))

if __name__ == "__main__":
    # Call the main function when the script is run as the main program
    main()
