# DQI Resource Estimation

Open a new python environment do an editable install using `pip install -e .`. 

Run the tests using `pytest -n auto .`

To reproduce the resource estimates in the paper, run 

```python
python rs_codes/opi_resource_estimates.py --algo zalka
```

To compare against resource estimates using Dialog based EEA, run

```python
python rs_codes/opi_resource_estimates.py --algo dialog
```

To generate a table of primitive operations that the DQI quantum circuit for OPI decomposes to, run 

```python
python dqi/rs_codes/opi_gate_cost_table.py --m 4095 --n 70 --b 12 
```
