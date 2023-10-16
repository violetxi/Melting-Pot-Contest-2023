import matplotlib.pyplot as plt


# Function to create bar plots with mean lines and vertical agent names inside the bars for focal_per_capita_return
def plot_with_vertical_labels_inside(df, title):
    # Set the positions and width for the bars
    pos = list(range(len(df['background_per_capita_return']))) 
    width = 0.25 

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))

    # Create bars for background_per_capita_return with mean line
    plt.bar(pos, 
            df['background_per_capita_return'], 
            width, 
            alpha=0.5, 
            color='blue', 
            label='background_per_capita_return') 
    
    bg_mean = df['background_per_capita_return'].mean()
    plt.axhline(y=bg_mean, color='blue', linestyle='--', linewidth=1.5)
    plt.text(0.95, bg_mean, '{:.2f}'.format(bg_mean), color='blue', va='center', ha="right", bbox=dict(facecolor="w",alpha=0.5),
             transform=ax.get_yaxis_transform())

    # Create bars for focal_per_capita_return with mean line
    bars = plt.bar([p + width for p in pos], 
            df['focal_per_capita_return'],
            width, 
            alpha=0.5, 
            color='red', 
            label='focal_per_capita_return') 

    focal_mean = df['focal_per_capita_return'].mean()
    plt.axhline(y=focal_mean, color='red', linestyle='--', linewidth=1.5)
    plt.text(0.95, focal_mean, '{:.2f}'.format(focal_mean), color='red', va='center', ha="right", bbox=dict(facecolor="w",alpha=0.5),
             transform=ax.get_yaxis_transform())

    # Labeling the focal_per_capita_return bars with vertical agent names
    for bar, agent_name in zip(bars, df['focal_player_names']):
        height = bar.get_height()
        ax.annotate(f'{agent_name[0]}',
                    xy=(bar.get_x() + bar.get_width() / 2, height/2),  # Positioning label in the middle of the bar
                    xytext=(0, 0),  # No offset
                    textcoords="offset points",
                    ha='center', va='center', rotation=90)  # Vertical text

    # Set the y axis label
    ax.set_ylabel('Returns')

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['Unnamed: 0'])

    # Adding the legend and showing the plot
    plt.legend(['Mean background_per_capita_return', 'Mean focal_per_capita_return
