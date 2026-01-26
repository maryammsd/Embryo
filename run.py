import os
import embryo
import cleavage
import math
import core
import test
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter


def traverse_directory(path):
    count = 0 
    for root, dirs, files in os.walk(path):
        print(f"Directory: {root}")
        dir_name = os.path.basename(root)
      
        #if dir_name != "D4":
        #    continue
        for file_name in files:
            print(f"  Directory Name: {dir_name}")
            embryos_data = {}
            parent_child = {} 
            timestamp_data = {}
            embryos_data, parent_child, timestamp_data = core.read_file(os.path.join(root, file_name),dir_name)
            cleavage_data = core.get_cleavage_info(embryos_data, parent_child, timestamp_data)
            if cleavage_data is None:
                print(f"No cleavage data found for {file_name} in {root}.")
                continue
            count += 1
            print(f"  Cleavage Data for directory {dir_name}: {cleavage_data}")
            cleavage_dict[dir_name] = cleavage_data

def depict_histogram(x_values, y_values, bin_size=10):
    plt.hist(x_values, bins=bin_size, alpha=0.7, label='Embryo Cleavage Longetivity')
    plt.hist(y_values, bins=bin_size, alpha=0.7, label='Parent Cleavage Longetivity')
    plt.xlabel(f"Longetivity (Hour) with median: {np.median(x_values):.2f} and mean: {np.mean(x_values):.2f}")
    plt.legend()
    plt.show()
    plt.savefig("cleavage_histogram.png")
    plt.close()

def depict_violion_plot(data_child:dict, data_parent:dict, file_name:str, title:str, xlabel:str, ylabel:str):

    categorized_data = list(data_child.keys())
    data_child = list(data_child.values())
    data_parent = list(data_parent.values())

    positions = np.arange(1, len(categorized_data) + 1)

    plt.figure(figsize=(12, 6))

   

    # depict violion plot for child and parent in the same canvas with different colors
    vp1 =  plt.violinplot(data_child, showmeans=True, positions= positions, widths= 0.5, showmedians=True, showextrema=True,)
    vp2 = plt.violinplot(data_parent, showmeans=True, positions= positions, widths= 0.5,  showmedians=True, showextrema=True)
    plt.title(title)
        
    # Color with transparency
    for body in vp1['bodies']:
        body.set_facecolor('blue')
        body.set_alpha(0.5)

    for body in vp2['bodies']:
        body.set_facecolor('red')
        body.set_alpha(0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legend_elements = [
    Patch(facecolor='blue', alpha=0.5, label='Child Longetivity'),
    Patch(facecolor='red', alpha=0.5, label='Parent Longetivity')
    ]
    plt.legend(handles=legend_elements)

    # define x-ticks
    x_ticks = []
    for i in range(len(categorized_data)):
        current_tick = categorized_data[i]
        previous_tick = str(int(math.pow(2,math.log2(int(current_tick)) - 1)) if i > 0 else "4")
        x_ticks.append(f"{previous_tick} -> {current_tick}")

    plt.xticks(range(1, len(categorized_data) + 1), x_ticks)  # Set x-axis labels
    plt.show()
    # save plot as png
    plt.savefig(file_name)
    plt.close()

def depict_scatter_plot(x_values, y_values):
    plt.scatter(x_values, y_values)
    plt.xlabel("Embryo Cleavage Longetivity")
    plt.ylabel("Parent Cleavage Longetivity")
    plt.title("Embryo Cleavage vs Parent Cleavage")

    # Regression line for Dataset 1
    slope1, intercept1, r1, p1, _ = stats.linregress(x_values, y_values)
    line1_x = np.array([min(x_values), max(x_values)])
    line1_y = slope1 * line1_x + intercept1
    plt.plot(line1_x, line1_y, color='blue', linestyle='--', 
         label=f'All data: r={r1:.3f}')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
    # save plot as png
    plt.savefig("cleavage_scatter.png")
    plt.close()

def depict_scatter_with_counts(x_values, y_values, levels):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    levels = np.array(levels)
    level_counts = Counter(levels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Scatter
    scatter = ax1.scatter(x_values, y_values, c=levels, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='Cleavage Level')
    ax1.set_xlabel("Embryo Cleavage Longetivity")
    ax1.set_ylabel("Parent Cleavage Longetivity")
    ax1.set_title("Embryo Cleavage vs Parent Cleavage")
    
    # Right plot: Bar chart of counts
    sorted_levels = sorted(level_counts.keys())
    counts = [level_counts[level] for level in sorted_levels]
    
    bars = ax2.bar(sorted_levels, counts, color='steelblue', edgecolor='black')
    
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel("Cleavage Level")
    ax2.set_ylabel("Count")
    ax2.set_title("Items per Cleavage Level")
    ax2.set_xticks(sorted_levels)
    
    plt.tight_layout()
    plt.savefig("scatter_with_counts.png")
    plt.show()
    plt.close()

def depict_scatter_plot_colored(x_values, y_values, levels):
    plt.figure()
    scatter = plt.scatter(x_values, y_values, c=levels, cmap='Spectral', alpha=0.7)
    plt.colorbar(scatter, label='Cleavage Level')

    # Convert to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    levels = np.array(levels)
    
    # Get unique levels and colormap
    unique_levels = sorted(set(levels))
    cmap = plt.cm.Spectral
    norm = plt.Normalize(min(levels), max(levels))
    
    # Draw regression line for each level
    for level in unique_levels:
        # Get data points for this level
        mask = levels == level
        x_level = x_values[mask]
        y_level = y_values[mask]
        
        # Skip if not enough points
        if len(x_level) < 2:
            continue

        # Calculate correlations for each level separately
        pearson_r, pearson_p = stats.pearsonr(x_level, y_level)
        spearman_r, spearman_p = stats.spearmanr(x_level, y_level)
        kendall_r, kendall_p = stats.kendalltau(x_level, y_level)

        # Calculate regression
        slope, intercept, r, p, _ = stats.linregress(x_level, y_level)
        
        # Create line points
        line_x = np.array([min(x_level), max(x_level)])
        line_y = slope * line_x + intercept
        
        # Get color matching the scatter points
        color = cmap(norm(level))
        
        # Plot regression line
        plt.plot(line_x, line_y, color=color, linestyle='--', linewidth=2,
                 label=f'Level {level}: r={pearson_r:.3f} p-value = {pearson_p:.2e}, ρ = {spearman_r:.4f} p-value = {spearman_p:.2e}, τ = {kendall_r:.4f} p-value = {kendall_p:.2e}, R² = {r**2:.4f}')

    plt.xlabel("Embryo Cleavage Longetivity")
    plt.ylabel("Parent Cleavage Longetivity")
    plt.title("Embryo Cleavage vs Parent Cleavage (Colored by Level)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='lower right', fontsize=8)
    plt.show()
    # save plot as png
    plt.savefig("cleavage_scatter_colored.png")
    plt.close()


def parent_child_special_age(x_values, y_values):
    # Filter data for special age range
    filtered_x_low = []
    filtered_y_low = []
    filtered_x_high = []
    filtered_y_high = []
    for x, y in zip(x_values, y_values):
        if 8 <= y <= 11:
            filtered_x_low.append(x)
            filtered_y_low.append(y)
        elif y > 11:
            filtered_x_high.append(x)
            filtered_y_high.append(y)

    pearson_r_low, pearson_p_low = stats.pearsonr(filtered_x_low , filtered_y_low )
    pearson_r_high, pearson_p_high = stats.pearsonr(filtered_x_high , filtered_y_high )
    plt.scatter(filtered_x_low, filtered_y_low, color='blue', label='8-11')
    plt.scatter(filtered_x_high, filtered_y_high, color='red', label='>11')
    plt.xlabel(f"Embryo Cleavage Longetivity: pearson r low={pearson_r_low:.3f} p={pearson_p_low:.2e}, high={pearson_r_high:.3f} p={pearson_p_high:.2e}")
    plt.ylabel("Parent Cleavage Longetivity ")
    plt.title("Special Age Range: Embryo vs Parent Cleavage")


    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.show()
    # save plot as png
    plt.savefig("cleavage_special_age_scatter.png")
    plt.close()


def perform_t_test(x_values, y_values, levels):
    # 2. Perform the Paired T-Test
    # ttest_rel compares two related samples
    t_stat, p_val = stats.ttest_rel(x_values, y_values)

    print(f"Paired T-test Result: t-stat = {t_stat:.3f}, p-value = {p_val:.3f}")

    # 3. Prepare data for visualization (Melt to 'Long Format')
    df = pd.DataFrame({
        'Parent': x_values,
        'Child': y_values,
        'Level': levels
    })
    df_melted = df.melt(id_vars=['Level'], value_vars=['Parent', 'Child'], 
                        var_name='Generation', value_name='Longevity')

    # 4. Visualization
    plt.figure(figsize=(10, 6))

    # Boxplot shows the distribution of each group
    sns.boxplot(data=df_melted, x='Generation', y='Longevity', palette='Set2', width=0.5)

    # Stripplot shows individual data points and their corresponding cleavage level
    sns.stripplot(data=df_melted, x='Generation', y='Longevity', hue='Level', 
                palette='viridis', size=8, alpha=0.8, dodge=True)

    # Annotate with P-value for 2026 standards
    y_max = df_melted['Longevity'].max()
    plt.text(0.5, y_max, f'Paired t-test p = {p_val:.6f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    plt.title('Longevity Comparison: Parent vs. Child')
    plt.ylabel('Longevity (Time Units)')
    plt.legend(title='Cleavage Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def depict_scatter_with_lowess(x_values, y_values):
    # 1. Prepare your data (replace with your actual data lists)
# Using placeholder data that mimics your stats (monotonic but non-linear)
    data = {
        'Parent_Count': x_values,
        'Child_Count': y_values
    }
    df = pd.DataFrame(data)

    # 2. Create the plot
    plt.figure(figsize=(10, 6))

    # sns.regplot with lowess=True shows the non-linear trend
    sns.regplot(x='Parent_Count', y='Child_Count', data=df,
                lowess=True, line_kws={'color': 'red', 'label': 'Monotonic Trend'},
                scatter_kws={'alpha': 0.6})

    # 3. Add labels and title
    plt.title('Relationship: Parent vs Child Counts', fontsize=14)
    plt.xlabel('Embryo Age', fontsize=12)
    plt.ylabel('Parent Age', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

def main_analysis(cleavage_dict):
    # x = embryo cleavage_timestamp and y = parent_cleavage_timestamp
    x_values = []
    y_values = []
    levels = [] 
    longetivity_level_child = {}
    longetivity_level_parent = {}   
    longetivity_by_time = {}
    for dir_name in cleavage_dict:
        cleavage_data = cleavage_dict[dir_name]
        current_level = 8
        # remove cleavage level less than 4
        cleavage_data = {k: v for k, v in cleavage_data.items() if v.level > 4 and v.level < 64}

        for child_id in cleavage_data:
            # check if it is the first child in cleavage dict with level 8 
            if child_id == list(cleavage_data.keys())[0]:
                current_level = 8
            print(f"Child ID in depiction: {child_id} with cleavage data {cleavage_data[child_id].cleavage_timestamp}, {cleavage_data[child_id].parent_cleavage_timestamp   }")
            cleavage_info = cleavage_data[child_id]
            if cleavage_info.level <= 4:
                continue
            level = cleavage_info.level
            if level not in longetivity_level_child:
                longetivity_level_child[level] = []
            if level not in longetivity_level_parent:
                longetivity_level_parent[level] = []
            longetivity_level_child[level].append(float(cleavage_info.cleavage_timestamp *  15 / 60))  # convert to hours
            longetivity_level_parent[level].append(float(cleavage_info.parent_cleavage_timestamp *  15 / 60))  # convert to hours
            # only consider longetivity between 25 to 60
            #if float(cleavage_info.cleavage_timestamp) > 25 and float(cleavage_info.cleavage_timestamp) < 60:
            #    if float(cleavage_info.parent_cleavage_timestamp) > 25 and  float(cleavage_info.parent_cleavage_timestamp) < 60 :
            if level == 16:
                x_values.append(float(cleavage_info.cleavage_timestamp *  15 / 60))
                y_values.append(float(cleavage_info.parent_cleavage_timestamp *  15 / 60))            
            if current_level not in longetivity_by_time:
                longetivity_by_time[current_level] = []
            longetivity_by_time[current_level].append(float(cleavage_info.cleavage_timestamp *  15 / 60))
            current_level += 1
            levels.append(cleavage_info.level)

    # Depict scatter plot
    depict_scatter_plot(x_values, y_values)

    # Depict scatter plot with lowess
    depict_scatter_with_lowess(x_values, y_values)

    # depict scatter plot with special age range
    parent_child_special_age(x_values, y_values)
    # Depict a scatter plot with regression line and different colors for levels
    depict_scatter_plot_colored(x_values, y_values, levels)
    depict_scatter_with_counts(x_values, y_values, levels)

    # Depict the histogram
    depict_histogram(x_values, y_values)

    # Depict the violion plot
    depict_violion_plot(longetivity_level_child,longetivity_level_parent,file_name="cleavage_violion.png", title="Distribution of parent and child longetivity in each level", xlabel="Cleavage Level", ylabel="Longetivity (Hour)")

    # Depict the violion plot by time
    #depict_violion_plot(longetivity_by_time,file_name="cleavage_violion_by_time.png", title="Distribution of child longetivity by time", #xlabel="Time (Timestamp)", ylabel="Longetivity (Timestamp)")

    # perform statistical tests
    # perform chi-test
    test.chi_test(longetivity_level_child, longetivity_level_parent)
    test.spearmanr_test(longetivity_level_child, longetivity_level_parent)
    test.kendalltau_test(longetivity_level_child, longetivity_level_parent)
    test.fisher_exact_test(longetivity_level_child, longetivity_level_parent)

    # Calculate average of x and y
    sum_x = 0
    sum_y = 0
    for x in x_values:
        sum_x += math.pow(x,2)
    for y in y_values:
        sum_y += math.pow(y,2)
    print(" average x_values:", {sum(x_values)/len(x_values)})
    print(" average y_values:", {sum(y_values)/len(y_values)})
    print(" variance x_values:", {sum_x/len(x_values)})
    print(" variance y_values:", {sum_y/len(y_values)})

    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(x_values, y_values)
    print(f"Pearson correlation: {correlation}, p-value: {p_value}")
    
    # Display correlation on the plot
    #plt.text(10, 180, f'Pearson r = {correlation:.4f}\np-value = {p_value:.4e}', 
    #         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def print_cleavage_data():
    for cleavage_data in cleavage_dict:
        for child_id in cleavage_data:
            cleavage_info = cleavage_data[child_id]
            print(f"Cleavage Info for Embryo ID {child_id}:")
            print(f"  Parent ID: {cleavage_info.parent_id}")
            print(f"  Cleavage Hour: {cleavage_info.cleavage_timestamp}")
            print(f"  Parent Cleavage Hour: {cleavage_info.parent_cleavage_timestamp}")
            print(f"  Level: {cleavage_info.level}")
            print(f"  Species Type: {cleavage_info.species_type}")
            print(f"  Embryo Info: {cleavage_info.embryo_info}")
            print("-----")

def save_to_csv(cleavage_dict):
    csv_file = "cleavage_data.csv"
    with open(csv_file, "w") as f:
        f.write("ChildCleavageLongetivity,ParentCleavageLongetivity,Level,EmbryoInfo,EmbryoID,ParentID\n")
        for dir_name in cleavage_dict:
            cleavage_data = cleavage_dict[dir_name]
            for child_id in cleavage_data:
                cleavage_info = cleavage_data[child_id]
                f.write(f"{cleavage_info.cleavage_timestamp},{cleavage_info.parent_cleavage_timestamp},{cleavage_info.level},{cleavage_info.embryo_info},{cleavage_info.embryo_id},{cleavage_info.parent_id}\n")
        print(f"Saved cleavage data to {csv_file}")


def refine_cleavage_data(cleavage_dict):
    # for each level, keep it for the embryo with maximum number of data points
    refined_cleavage_dict = {}
    for dir_name in cleavage_dict:
        data = cleavage_dict[dir_name]
        combo_counts = {}
        for item in data:
            embryoInfo = data[item]
            if embryoInfo.level not in combo_counts:
                combo_counts[embryoInfo.level] = 0
            combo_counts[embryoInfo.level] += 1

        for item in data:
            embryoInfo = data[item]
            # only keep the embryoInfo if its count matches the level
            print(f"Embryo ID: {embryoInfo.embryo_id}, Level: {embryoInfo.level}, Count: {combo_counts[embryoInfo.level]}")
            if combo_counts[embryoInfo.level] == embryoInfo.level:
                if dir_name not in refined_cleavage_dict:
                    refined_cleavage_dict[dir_name] = {}
                id_ = embryoInfo.embryo_id
                refined_cleavage_dict[dir_name][id_] = embryoInfo

    return refined_cleavage_dict

def main():
    print("Let's run the application!")
    traverse_directory("./tracking")
    refined_cleavage_data = {}
    #refined_cleavage_data = refine_cleavage_data(cleavage_dict=cleavage_dict)
    main_analysis(cleavage_dict=cleavage_dict)
    save_to_csv(cleavage_dict=cleavage_dict)

if __name__ == "__main__":
    cleavage_dict = {}
    x_max = 30
    x_min = 7.5
    y_min = 2.5
    y_max = 20
    main()
   
            
