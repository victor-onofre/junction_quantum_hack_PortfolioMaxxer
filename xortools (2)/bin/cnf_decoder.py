import stim
import numpy as np
import pysat
# pip install python-sat[aiger,approxmc,cryptosat,pblib]
import pysat.examples.rc2
import pysat.formula
import itertools
import numpy as np
from tqdm.auto import tqdm
import time
import argparse
import gzip
from bp import load_instance_from_csv, load_instance_from_tsv, sample_shots
import subprocess
import signal

def timeout_sigterm(N, binary_name):
  # Start the process
  process = subprocess.Popen(binary_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  try:
    # Wait for N seconds
    time.sleep(N)
    # Send SIGTERM signal to the process
    process.send_signal(signal.SIGTERM)
    # Wait for the process to finish
    stdout, _ = process.communicate()
    process.wait()
    # Return the output
    return stdout.decode().split('\n')
  except subprocess.TimeoutExpired:
      # If the process hasn't finished within N seconds, kill it forcefully
      process.kill()
      stdout, _ = process.communicate()
      return stdout.decode().split('\n')

class CNFDecoder:
  def __init__(self, instance, ell, verbose=False):
    self.num_bits = 0
    self.num_detectors = instance['n']
    assert instance['p'] == 2
    self.errors = []
    for constraint in instance['constraints']:
      # Each constraint gives an error mechanism in DQI
      probability = ell/instance['m']
      detectors = constraint['variables']
      error = {
        'probability': probability,
        'detectors': detectors,
      }
      self.errors.append(error)

    self.errors_by_detector = [[] for _ in range(self.num_detectors)]
    for ei, error in enumerate(self.errors):
      for detector in error['detectors']:
        self.errors_by_detector[detector].append(ei)

    self.error_costs = np.array([1 for error in self.errors])
    self.wcnf = pysat.formula.WCNF()
    self.error_inclusions = [self.new_bit() for ei in range(len(self.errors))]
    self.is_flipped = [self.new_bit() for detector in range(instance['n'])]

    for detector in tqdm(range(instance['n'])):
      local_errors = self.errors_by_detector[detector]
      if len(local_errors) == 1:
        parity = self.error_inclusions[local_errors[0]]
      else:
        assert len(local_errors) >= 2
        parity = self.XOR(self.error_inclusions[local_errors[0]], self.error_inclusions[local_errors[1]])
        for i in range(2, len(local_errors)):
          parity = self.XOR(parity, self.error_inclusions[local_errors[i]])
      self.AddEq(self.is_flipped[detector], parity)

    for ei in range(len(self.errors)):
      cost = self.error_costs[ei]
      self.wcnf.append([self.Not(self.error_inclusions[ei])], weight=cost)

  def XOR(self, a, b):
    result = self.new_bit()
    x = [a, b, result]
    for bs in itertools.product([0, 1], repeat=3):
      if ((bs[0] + bs[1])%2) == bs[2]:
        continue
      # Must make sure x != bs
      clause = []
      for i in range(3):
        # x[i] != bs[i]
        bit = x[i]
        if bs[i]:
          bit = self.Not(bit)
        clause.append(bit)
      self.wcnf.append(clause)
    return result

  def Not(self, bit):
    return -bit

  def new_bit(self):
    index = self.num_bits + 1
    self.num_bits += 1
    return index

  def AddEq(self, a, b):
    self.wcnf.append([a, -b])
    self.wcnf.append([-a, b])

  def decode(self, syndromes, errors=None, verbose=False, write_model_file_name_and_exit=None):
    num_shots = syndromes.shape[0]

    # Decode with SAT solver
    num_errors = 0
    start_time = time.time()
    for shot in range(num_shots):
      syndrome = syndromes[shot]
      if write_model_file_name_and_exit is not None:
        wcnf = self.wcnf.copy()
        for detector, val in enumerate(syndrome):
          a = self.is_flipped[detector]
          if not val:
            a = self.Not(a)
          wcnf.append([a])
        with open(write_model_file_name_and_exit, 'wt') as outfile:
          wcnf.to_fp(outfile)
        exit(0)
        # outputs = timeout_sigterm(100, '/usr/local/google/home/shutty/projects/satsolvers/Loandra/bin/loandra_static')
        # assert outputs[-1][0] == 'v'
        # solution = map(int, outputs[-1].split()[1])
      else:
        with pysat.examples.rc2.RC2Stratified(
            self.wcnf,
            solver='glucose4',
            incr=True,
            blo='full', minz=True,
            verbose=10 if verbose else 0
        ) as rc2:
          for detector, val in enumerate(syndrome):
            a = self.is_flipped[detector]
            if not val:
              a = self.Not(a)
            rc2.add_clause([a])
          model = rc2.compute()
          if verbose:
            print(rc2.cost)
          solution = []
          for ei, bit in enumerate(model):
            if ei >= len(self.errors):
              break
            if bit == -self.error_inclusions[ei]:
              solution.append(0)
            elif bit == self.error_inclusions[ei]:
              solution.append(1)
            else:
              raise ValueError('Invalid model')

      solution = np.array(solution)
      if verbose:
        print(solution)

      # Check that the solution is valid (gives the correct syndrome)
      # if these checks do not pass, then the solver returned an invalid
      # assignment
      for detector in range(self.num_detectors):
        activated = sum(
          solution[ei]
          for ei in self.errors_by_detector[detector]
        ) % 2
        detection = syndrome[detector]
        assert activated == detection

      # Check if we got the right error
      error = False
      if errors is not None:
        difference = (solution - errors[shot]) % instance['p']
        print(f'error weight = {difference.astype(bool).sum()}')
        error = bool((solution != errors[shot]).any())
        if error:
          num_errors += 1
          if verbose:
            print('error')

      total_cost = np.dot(solution, self.error_costs)
      if verbose:
        print(f'total cost = {total_cost}')
      current_time = time.time()
      if verbose:
        print(f'done in {current_time-start_time} seconds')
      return {
        'error': error,
        'num_bits_flipped': int(np.sum(solution)),
        'total_time': current_time-start_time,
      }
      

if __name__ == '__main__':
  import json
  parser = argparse.ArgumentParser('Decode some shots using maxSAT solver')
  parser.add_argument('--csv', required=False, type=str, default=None, help='path to a .csv instance file')
  parser.add_argument('--tsv', required=False, type=str, default=None, help='path to a .tsv instance file')
  parser.add_argument('--sample-num-shots', required=True, type=int,
                      default=None,
                      help='number of shots to sample from the circuit')
  parser.add_argument('--sample-seed', required=False, type=int,
                      default=None,
                      help='seed used to sample shots (if sampling)')
  parser.add_argument('--ell', required=True, type=int,
                      default=None,
                      help='number of errors to add in each shot')
  parser.add_argument('--write-model-file', type=str, default=None,
                      help='.wcnf file to save a shot to (exits afterwards)')
  parser.add_argument('--verbose', action='store_true', default=False,
                      help='show verbose solver output and incremental stats '
                           'during decoding')
  parser.add_argument('--only-decode-shot', type=int, default=None,
                      action='append',
                      help='If present, only decode this specific shot index. '
                      'Can use multiple times to decode more than 1 specific shot.')
  args = parser.parse_args()
  if args.sample_num_shots is not None:
    np.random.seed(args.sample_seed)

  num_model_sources = int(args.tsv is not None) + int(args.csv is not None)
  if num_model_sources != 1:
    raise ValueError('Must supply either --csv or --tsv')
  if args.tsv is not None:
    instance = load_instance_from_tsv(args.tsv)
  if args.csv is not None:
    instance = load_instance_from_csv(args.csv)

  decoder = CNFDecoder(instance, args.ell, verbose=args.verbose)
  syndromes, errors = sample_shots(instance, args.ell, args.sample_num_shots)
  if args.only_decode_shot is not None:
    trimmed_syndromes = []
    trimmed_errors = []
    for shot in args.only_decode_shot:
      trimmed_syndromes.append(syndromes[shot])
      trimmed_errors.append(errors[shot])
    syndromes = np.array(trimmed_syndromes)
    errors = np.array(trimmed_errors)
  results = decoder.decode(syndromes, errors=errors, verbose=args.verbose,
                           write_model_file_name_and_exit=args.write_model_file)
  results['only_decode_shot'] = args.only_decode_shot
  results['sample_num_shots'] = args.sample_num_shots
  results['sample_seed'] = args.sample_seed
  print(json.dumps(results))
