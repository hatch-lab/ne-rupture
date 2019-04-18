def colorize(color, string):
  """
  Used to print colored messages to terminal

  Arguments:
    color string The color to print
    string string The message to print

  Returns:
    A formatted string
  """
  colors = {
    "red": "31",
    "green": "32",
    "yellow": "33", 
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37"
  }

  return "\033[" + colors[color] + "m" + string + "\033[0m"