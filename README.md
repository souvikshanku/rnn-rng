# rnn-rng

Random number generator powered by LSTM. Some RNGs are less random than the others. Needless to say, this is one of the less random ones! 😅

## Example Usage

```bash
git clone https://github.com/souvikshanku/rnn-rng.git
cd rnn-rng

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data and train the model
python train.py
```

Now use the trained model to generate random intergers in $[0, 99]$.

```python
from utils import generate_random_number
print(generate_random_number())
```
