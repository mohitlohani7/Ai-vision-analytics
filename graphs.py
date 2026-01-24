import matplotlib.pyplot as plt

def generate_class_graph(class_count, save_path):
    if not class_count:
        return

    classes = list(class_count.keys())
    counts = list(class_count.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='skyblue')
    
    plt.xlabel("Object Class", fontsize=12)
    plt.ylabel("Unique Count", fontsize=12)
    plt.title("Unique Object Detection Count", fontsize=14)
    
    # Bars ke upar number likhne ke liye
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()