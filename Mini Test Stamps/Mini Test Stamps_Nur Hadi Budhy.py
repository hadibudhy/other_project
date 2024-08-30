def is_prime(num):
  """Checks if a number is prime."""
  if num <= 1:
    return False
  if num <= 3:
    return True
  if num % 2 == 0 or num % 3 == 0:
    return False
  i = 5
  while i * i <= num:
    if num % i == 0 or num % (i + 2) == 0:
      return False
    i += 6
  return True

def get_replacement(num):
  """Returns the replacement text for a number based on divisibility rules."""
  if num % 15 == 0:
    return "FooBar"
  elif num % 3 == 0:
    return "Foo"
  elif num % 5 == 0:
    return "Bar"
  else:
    return str(num)

# Create a list of numbers from 1 to 100
numbers = list(range(1, 101))

# Filter out prime numbers and replace numbers according to rules
filtered_numbers = [get_replacement(num) for num in numbers[::-1] if not is_prime(num)]

# Print the filtered numbers horizontally
print(" ".join(filtered_numbers))