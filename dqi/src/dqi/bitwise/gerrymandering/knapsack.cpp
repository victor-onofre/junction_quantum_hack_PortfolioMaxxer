#include "knapsack.h"

#include <assert.h>

#include "problem_def.h"

namespace {

probability_distribution_t construct_expected_future_gain(
    int m, int s, int budget,
    const std::vector<std::vector<std::vector<std::pair<int, probability_t>>>>&
        table,
    const std::vector<probability_t>& p_s) {
  if (m == 0) {
    return probability_distribution_t{1};
  }
  std::vector<probability_t> probs;
  for (; m; m--) {
    s = table[m][s][budget].first;
    if (s == -1) return probability_distribution_t();
    assert(s <= budget);
    budget -= s;
    probs.push_back(p_s[s]);
  }
  return distribution_from_probabilities(probs);
}
std::vector<std::vector<std::vector<std::pair<int, probability_t>>>>
create_lookahead_table(const GerrymanderingInstance& gi) {
  std::vector<std::vector<std::vector<std::pair<int, probability_t>>>>
      lookahead(gi.m + 1,
                std::vector<std::vector<std::pair<int, probability_t>>>(
                    gi.b + 1, std::vector<std::pair<int, log_probability_t>>(
                                  gi.budget + 1)));
  for (int i = 1; i <= gi.m; i++) {
    for (int s = gi.b; s >= 0; s--) {
      for (int budget = 0; budget <= gi.budget; budget++) {
        lookahead[i][s][budget] = std::pair<int, probability_t>(-1, 0);
        if (s < gi.b) lookahead[i][s][budget] = lookahead[i][s + 1][budget];
        if (s <= budget) {
          auto tmp = lookahead[i - 1][s][budget - s];
          tmp.second += gi.p_s[s];
          tmp.first = s;
          if (tmp.second > lookahead[i][s][budget].second)
            lookahead[i][s][budget] = tmp;
        }
      }
    }
  }
  return lookahead;
}

MaximizationResult solve(const GerrymanderingInstance& gi, bool fast) {
  std::vector<std::vector<std::vector<std::pair<int, probability_t>>>>
      lookahead;
  if (!fast) {
    lookahead = create_lookahead_table(gi);
  }
  std::vector<std::vector<knapsack::Result>> table[2];
  probability_distribution_t aux;
  for (int i = 0; i < 2; i++) {
    table[i].resize(gi.b + 1);
    for (int s = 0; s <= gi.b; s++) table[i][s].resize(gi.budget + 1);
  }
  for (int s = 0; s <= gi.b; s++)
    for (auto& r : table[1][s]) r.clear();
  for (int i = 1; i <= gi.m; i++) {
    table[0].swap(table[1]);
    for (int s = 0; s <= gi.b; s++) {
      for (int budget = 0; budget <= gi.budget; budget++) {
        table[1][s][budget].clear();
      }
    }
    for (int s = 0; s <= gi.b; s++) {
      for (int budget = s; budget <= gi.budget; budget++) {
        if (table[0][s][budget - s].is_valid(i - 1)) {
          table[1][s][budget] = table[0][s][budget - s].convolve(s, gi.p_s[s]);
          assert(table[1][s][budget].probs.size() == i + 1);
        }
      }
    }
    for (int s = gi.b - 1; s >= 0; s--) {
      for (int budget = 0; budget <= gi.budget; budget++) {
        probability_distribution_t future_gain;
        if (!fast) {
          future_gain = construct_expected_future_gain(
              gi.m - i, s, gi.budget - budget, lookahead, gi.p_s);
        }
        if (table[1][s + 1][budget].is_better_than(table[1][s][budget],
                                                   gi.target, future_gain)) {
#ifdef VERBOSE
          std::cerr << "\tat i=" << i << " s=" << s << " budget=" << budget
                    << " replace " << table[1][s][budget] << " with "
                    << table[1][s + 1][budget] << std::endl;
// std::cerr << table[1][s][budget].potential(gi.target, w) << " " << table[1][s
// + 1][budget].potential(gi.target, w) << " | w=" << w << std::endl;;
#endif
          table[1][s][budget] = table[1][s + 1][budget];
        }
      }
    }
#ifdef VERBOSE
    for (int s = 0; s <= gi.b; s++) {
      for (int budget = 0; budget <= gi.budget; budget++) {
        std::cerr << table[1][s][budget] << " ";
        assert(!table[1][s][budget].is_valid(i) ||
               table[1][s][budget].choices.size() == i);
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
#endif
  }
  return table[1][0][gi.budget].to_maximization_result(gi.target);
}
};  // namespace

namespace knapsack {
///////////////////////// Result Handling
//////////////////////////////////////////

Result Result::EmptyResult() {
  Result r;
  r.probs.push_back(1);
  return r;
}

probability_t Result::ccdf(int target) const {
  if (target >= probs.size()) return 0;
  return std::accumulate(probs.begin() + target, probs.end(), (probability_t)0);
}

probability_t Result::potential(int target, probability_t w) const {
  auto ret = ccdf(target);
  probability_t wt = w;
  for (int i = std::min(target, (int)probs.size()) - 1; i >= 0; i--) {
    ret += wt * probs[i];
    wt *= w;
  }
  return ret;
}

bool Result::is_better_than(const Result& other, int target,
                            const probability_distribution_t& future) const {
  if (probs.size() != other.probs.size()) {
    return probs.size() > other.probs.size();
  }
  if (!future.empty() && target < probs.size() + future.size() - 1) {
    auto u = ::convolve(probs, future);
    auto v = ::convolve(other.probs, future);
    auto p1 = std::accumulate(u.begin() + target, u.end(), (probability_t)0);
    auto p2 = std::accumulate(v.begin() + target, v.end(), (probability_t)0);
    if (p1 != p2) {
      return p1 > p2;
    }
  }
  auto f1 = ccdf(target), f2 = other.ccdf(target);
  if (f1 != f2) {
    return f1 > f2;
  }
  for (int i = target - 1; i >= 0; i--) {
    probability_t mine = (i < probs.size()) ? probs[i] : 0;
    probability_t theirs = (i < other.probs.size()) ? other.probs[i] : 0;
    if (mine != theirs) {
      return mine > theirs;
    }
  }
  return false;
}

MaximizationResult Result::to_maximization_result(int budget) const {
  return MaximizationResult{ccdf(budget), choices};
}

Result Result::convolve(int s, probability_t p) const {
  Result ret;
  ret.probs = ::convolve(probs, p);
  ret.choices = choices;
  ret.choices.push_back(s);
  return ret;
}

void Result::clear() {
  choices.clear();
  probs.clear();
  probs.push_back(1);
}

bool Result::is_valid(int m) const {
  return probs.size() == m + 1 && choices.size() == m;
}

//////////////////////////// Maximization ////////////////////////////////
MaximizationResult maximize_probability(const GerrymanderingInstance& gi,
                                        bool fast) {
  return solve(gi, fast);
}
};  // namespace knapsack
