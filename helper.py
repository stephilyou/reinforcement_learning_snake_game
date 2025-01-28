# helper.py

def save_high_score(score, File):
    """Save the high score to a file."""
    with open(File, "w") as file:
        file.write(str(score))

def reset_high_score(File):
    """Save the high score as 0 to a file."""
    with open(File, "w") as file:
        file.write(str(0))
    return 0