#!/usr/bin/env python
"""\
Range Transferable Vote
Copyright (C) 2011,  Louis G. "Ted" Stern

For more information, see the README for this project.
"""
# -------- BEGIN cut and paste line for online interpreters --------
#
# For faster reverse sorting (in 2.4+):
from operator import itemgetter
from textwrap import fill, dedent
import re, os, sys

# Default maximum range/score
DEFAULT_MAX_SCORE = 10

DEFAULT_N_SCORE = DEFAULT_MAX_SCORE + 1

# Default number of seats in a multi-winner election
DEFAULT_NSEATS = 7

# Utility function to sort a dictionary in reverse order of values
def reverse_sort_dict(d):
    return sorted(d.iteritems(), key=itemgetter(1), reverse=True)

qtypes = ['hare', 'droop', 'droop-fractional', 'hagenbach-bischoff']

# Set up for Hare or Droop quotas:
def calc_quota(n, nseats=DEFAULT_NSEATS, qtype='droop'):
    # Hare quota = Nvotes / Nseats
    # Droop quota = int(Nvotes / (Nseats + 1)) + 1
    # Droop fractional = Droop quota with Nvotes*Nseats votes, then
    #                    dividing by Nseats.
    # Hagenbach-Bischoff = Nvotes / (Nseats + 1)

    fn = float(n)
    fs = float(nseats)
    fsp1 = fs + 1.0

    # We implement a CASE switch construction using a dict:
    return {'droop':              (float(int(fn/fsp1)) + 1.0),
            'hare':               (fn/fs),
            'droop-fractional':   ((float(float(int(fn*fs/fsp1)) + 1.0))/fs),
            'hagenbach-bischoff': (fn/fsp1)}[qtype]

class Ballot(dict):
    def __init__(self,csv_string='',cand_list=[]):
        # Parse the csv_string
        scores = []
        for i, v in enumerate(csv_string.rstrip().split(',')):
            if v:
                intv = int(v)
                if intv:
                    scores.append((cand_list[i],intv))
        # Now initialize with the list of 2-tuples
        dict.__init__(self,scores)

        self.rescale = 1.0

class Election(object):

    def __init__(self,
                 ballots=[],
                 candidates=set([]),
                 csv_input=None,
                 csv_output=None,
                 qtype='droop',
                 max_score=DEFAULT_MAX_SCORE,
                 nseats=DEFAULT_NSEATS):
        "Initialize from a list of ballots or a CSV input file"

        # Number of seats to fill:
        self.nseats = nseats

        # Quota type
        self.qtype = qtype
        if qtype not in qtypes:
            print "Error, qtype not recognized"
            sys.exit(1)

        # ------------------------------------------------------------
        # Absorb ballots, from input list and/or from file or stdin

        if ballots:
            self.ballots = ballots
        else:
            self.ballots = []

        self.candidates = candidates
        if csv_input:
            if csv_input == '-':
                self.csv_ballots(stdin=True)
            else:
                self.csv_ballots(filename=csv_input)

        # Initialize lists and sets of candidates:

        self.seated = set([])
        self.ordered_seated = []
        self.standing = self.candidates
        self.ordered_candidates=sorted(self.candidates)

        # Maximum Range score:
        self.max_score = max_score
        self.n_score = self.max_score + 1

        if csv_output:
            if csv_output == '-':
                self.csv_output = sys.stdout
            else:
                self.csv_output = open(csv_output, 'w')

        # Count the number of votes
        self.nvotes = len(self.ballots)

        # Calculate quota
        self.quota = calc_quota(self.nvotes,
                                self.nseats,
                                qtype=self.qtype)

        # Set up initial line of CSV output file:
        # Format is
        #
        # | Cand1 | Cand2 | ... | CandX | Winner name
        # +-------+-------+-----+-------+------------
        # |       |       |     |       |

        quota_string = "%s quota = %g out of %g\n" % \
            ( self.qtype.capitalize(),
              self.quota,
              self.nvotes )

        self.csv_lines = [','.join(self.ordered_candidates + [quota_string])]


        fmax = float(self.max_score)
        self.beta = [float(i)/fmax for i in range(self.n_score)]

        return

    def csv_ballots(self,
                    filename=None,
                    stdin=False):
        "Read ballots from a csv file.  First line is names of candidates."
        if stdin:
            f = sys.stdin
        else:
            f = open(filename,'r')

        # List of candidate names in the first line:
        keys = f.readline().rstrip().split(',')
        self.candidates.update(set(keys))

        # Following lines are the ballots:
	# A completely empty ballot could be construed as a vote for
	# "None of the Above".  Depending on how votes are counted, this could
	# force a run-off.
        for line in f:
            ballot = Ballot(line,keys)
	    self.ballots.append(ballot)

        if not stdin:
            f.close()
        return

    def compute_totals(self, factors, winner=None):
        """\
Loop over ballots, finding total reweighted votes at each rating level.

As part of ballot loop, first adjust rescale factor if ballot had the
previous winner with non-zero score.

Check whether ballot has any standing candidates.

If so, accumulate totals and locksums at each score level for each
standing candidate, and keep track of weighted total vote.
"""

        totals = [dict([(c,0.0) for c in self.standing])]
        locksums = [dict([(c,0.0) for c in self.standing])]
        total_vote = 0.0

        # Initialize dicts for each rating level.  We already
        # initialized above for score==0, but it is never used.
        for i in xrange(self.max_score):
            totals.append(dict([(c,0.0) for c in self.standing]))
            locksums.append(dict([(c,0.0) for c in self.standing]))

        # In a single loop over the ballots, we
        #
        # a) Rescale the ballot using the factor from the previous winner,
        #    if applicable (i.e. if this is not the first total calculation).
        #
        # b) Accumulate totals and locksums for each score level using
        #    the current rescale factor (after change above).
        #
        # "total_vote" is not used, but is accumulated as a check against
        # vote count.
        #
        for ballot in self.ballots:
            # Rescale ballot using factor from previous winner
            if winner:
                if ballot.has_key(winner):
		    ballot.rescale *= factors[ballot[winner]]

            rescale = ballot.rescale
            if rescale > 0.0:
                standing = set(ballot.keys()) & self.standing
                n = len(standing)
                if n > 0:
                    total_vote += rescale
                    for c in standing:
                        score = ballot[c]
                        totals[score][c] += rescale
                        if n == 1:
                            locksums[score][c] += rescale

        return totals, locksums, total_vote

# For keeping track of running totals in a Comma Separated Variable file
# that could be imported into a spreadsheet ...
    def print_running_total(self, ordered_scores):
        """Print CSV line of running total"""
        # This creates a "<formatted score> (<position>)" label for
        # each standing candidate.
        labels = {}
        for i, pair in enumerate(ordered_scores):
            c, score = pair
            labels[c] = "%15.5f (%d)" % (score, i+1)

        return ','.join([labels.get(c,'')
                         for c in self.ordered_candidates])

    def weighted_scoresum(self, totals, locksums):
        "Return weighted Range winner, weighted Range score and winner's locksum"

        # Candidates we're calculating totals for:
        standing = totals[1].keys()

        # Initial score sums for each candidate:
        total = dict([(c,0.0) for c in standing])
        locksum = dict([(c,0.0) for c in standing])

        # For each ratings level, weight the increment to the score sum
	# by the normalized score coefficient, beta:
        for score in xrange(1,self.n_score):
	    for c in standing:
		total[c] += self.beta[score] * totals[score][c]
		locksum[c] += self.beta[score] * locksums[score][c]

        # create a list of the score sum totals as (candidate, scoresum)
	# pairs, in reverse order of scoresum
	ordered_scores = reverse_sort_dict(total)

        # We now know the winner of this round:
	(winner, win_score) = ordered_scores[0]

	# ... and the corresponding locksum:
	lockval = locksum[winner]

	tied_scores = dict([(cand,score)
			    for cand, score in total.iteritems()
			    if score == win_score])

	csv_line = self.print_running_total(ordered_scores)

        # Check for tied winning scores:
	# (not doing anything at this point ...)
        if len(tied_scores) > 1:
            print "\nUh-oh!  There is a tie!"
            print "Tied candidates:"
            for c, score in tied_scores.iteritems():
                print "\t%s: %g" % (c, score)

	if (win_score >= self.quota):
	    csv_line += ",Seating %s; Locksum = %.5g\n" % (winner, lockval)
	    self.csv_lines.append(csv_line)
	else:
	    csv_line += ",Seating %s; Quota not reached\n" % winner
	    self.csv_lines.append(csv_line)

        return winner, win_score, lockval

    def run_election(self,
                     verbose=True,
                     debug=False,
                     terse=False):
        "Run the election."

        # make output extremely terse, if selected
        if terse:
            verbose = False

        if debug:
            verbose = True
            terse = False

        # Initiale rescaling factor and winner
	factors = [1.0 for i in xrange(self.n_score)]
        winner = None
        eps = 1.e-9
        n_seated = len(self.seated)
        n_needed = self.nseats - n_seated
        n_standing = len(self.standing)

        vote_count = float(self.nvotes)

        # Main loop:
        for i in xrange(self.nseats):

            # Calculate weighted totals and locksums.
            #
            # To avoid multiple loops through the ballots,
            # the rescaling for the previous winner's
            # rescale factor is done in the same loop.
            #
            # NB: Since we're rescaling ballots from the previous
            # iteration, total_votes is returned as the total number of
            # rescaled ballots before removing the new winner.
            #
            totals, locksums, total_vote = self.compute_totals(factors,
                                                               winner=winner)
            if not terse:
                print "total_vote = ", total_vote
                print "vote_count = ", vote_count

            # Given the totals and locksums for each approval level,
            # get the Bucklin winner, winner's Bucklin score and Locksum
            (winner,
             win_score,
             locksum) = self.weighted_scoresum(totals, locksums)

            # fraction used up = (Q-L)/(T-L),
            # constrained to lie between 0.0 and 1.0
            used_up_fraction = \
                max(self.quota - locksum, 0.0) / \
                max(max(win_score, self.quota) - locksum, eps)

            factors = [1.0 - self.beta[i] * used_up_fraction
                       for i in xrange(self.n_score)]

            vote_count -= min(max(locksum, self.quota),win_score)

            self.seated.add(winner)
            self.ordered_seated.append((winner,
                                        win_score,
                                        locksum,
                                        used_up_fraction))
            self.standing.remove(winner)

            n_seated += 1
            n_needed -= 1
            n_standing -= 1

            if not terse:

                print "Candidate %s seated in position %i" % ( winner,
                                                               n_seated), \
                    ", Score sum = %.5g" % win_score, \
                    ", Quota = %.5g" % self.quota, \
                    ", Locksum = %.5g" % locksum, \
                    ", Score used = %3.2f%%" % (used_up_fraction * 100.0), \
                    "\n"

        print "Winning set in order seated =",
        print "{" + ','.join([self.ordered_seated[i][0]
                              for i in range(self.nseats)]) + "}"

        print "Leftover vote =", vote_count

        # Write CSV output
        if self.csv_output == sys.stdout:
            print ""
            print "Begin CSV table output:"
            print "------8< cut here 8<---------"

        self.csv_output.writelines(self.csv_lines)

        if self.csv_output == sys.stdout:
            print "------8< cut here 8<---------"
            print "End CSV table output:"

        return

# -------- END cut and paste line for online interpreters --------
"""
If you don't have a python interpreter, you can run the code above
via the web, using

   http://ideone.com

Select Python from the left sidebar.

Cut and paste everything from from the "BEGIN cut and paste line" to
"END cut and paste line", and insert it into the source code textarea.

In the same textarea, following the source you've just cut and pasted
above, enter the appropriate input to run your example.  To run the
june2011.csv input, for example, you enter the following two statements:


election = Election(nseats=9,
                    max_score=9,
                    csv_input='-',
                    csv_output='-',
                    qtype='droop')

election.run_election()

Click where it says "click here to enter input (stdin) ...", and paste
in lines from the june2011.csv file.

Then click on the Submit button on the lower left.
"""

if __name__ == "__main__":
    from optparse import OptionParser

    usage="""%prog \\
            [-n|--nseats NSEATS] \\
            [-q|--quota-type QTYPE] \\
            [-i|--csv-input INPUT_FILENAME.csv] \\
            [-o|--csv-output OUTPUT_FILENAME.csv] \\
            [-v|--verbose] \\
            [-D|--debug]

%prog is a script to run Range Transferable Voting (RTV)
to implement a Proportional Representation (PR) election, using a set of
tabulated ballots with integer ratings for each candidate.

The Comma Separated Variable format is assumed to be in the form

	name1,name2,name3,...
        ,,,,,9,,,6,,7,,,2,...
        ,,9,8,7,6,1,2,0,...

with the list of candidates on the first line, and non-zero scores
for the respective candidates as ballots on following lines.
"""
    version = "Version: %prog 0.1"

    parser = OptionParser(usage=usage, version=version)

    parser.add_option('-n',
                      '--nseats',
                      type=int,
                      default=7,
                      help=fill(dedent("""\
                      Number of winning seats for election.  [Default: 7]""")))

    parser.add_option('-m',
                      '--max-score',
                      type=int,
                      default=5,
                      help=fill(dedent("""\
                      Maximum score.  [Default: %d]""" % DEFAULT_MAX_SCORE )))

    parser.add_option('-q',
                      '--quota-type',
                      type='string',
                      default='droop',
                      help=fill(dedent("""\
                      Quota type used in election.  'hare' = Hare =
                      Number of ballots divided by number of seats.
                      'droop' = Droop = Nballots /(Nseats + 1) + 1, dropping
                      fractional part.  'droop-fractional' =
                      (Nseats*Nballots)/(Nseats+1) + 1, drop fractional part,
                      then divide by Nseats.  It reduces to Droop when Nseats
                      is one. 'hagenbach-bischoff' = Nballots / (Nseats + 1).
                      [Default: droop]""")))

    parser.add_option('-i',
                      '--csv-input',
                      type='string',
                      default='-',
                      help=fill(dedent("""\
                      Filename of comma-separated-variable (csv) file
                      containing ballots.  Use hyphen ('-') to take
                      input from stdin.  [Default: -]""")))

    parser.add_option('-o',
                      '--csv-output',
                      type='string',
                      default='-',
                      help=fill(dedent("""\
                      Filename of comma-separated-variable (csv) file
                      to receive table of election results.
                      '.csv' extension can be included, but it will
                      be added if not present.
                      Use hyphen ('-') to direct output to stdout.
                      [Default: -]""")))

    parser.add_option('-v',
                      '--verbose',
                      action='store_true',
                      default=False,
                      help="Turn on verbose mode printout.  [Default:  False]")

    parser.add_option('-t',
                      '--terse',
                      action='store_true',
                      default=False,
                      help="Make printout even less verbose.  [Default:  False]")

    parser.add_option('-D',
                      '--debug',
                      action='store_true',
                      default=False,
                      help="Turn on debug mode printout.  [Default:  False]")

    opts, args = parser.parse_args()

    if opts.quota_type not in qtypes:
        print "\nError, argument to --quota-type must be one of", \
            ', '.join(["'%s'" % q for q in qtypes])
        parser.print_help()
        sys.exit(1)

    if (opts.nseats < 1):
        print "\nError, --nseats argument must be a positive integer\n"
        parser.print_help()
        sys.exit(1)

    csv_input = opts.csv_input
    csv_output = opts.csv_output
    if (csv_input == "-"):
        print "Reading CSV input from stdin\n\n"
    else:
        if not os.path.isfile(csv_input):
            print "\nError, %s file does not exist\n" % csv_input
            parser.print_help()
            sys.exit(1)

        ext = os.path.splitext(csv_input)[1]

        if not ext:
            csv_input += '.csv'
            ext = '.csv'
        elif ((ext != '.csv') and (ext != '.CSV')):
            print "\nError, %s file does not have .csv or .CSV extension\n" % csv_input
            parser.print_help()
            sys.exit(1)

    if (csv_output == "-"):
        print "Writing CSV input to stdout\n\n"
    else:

        ext = os.path.splitext(csv_output)[1]

        if not ext:
            csv_output += '.csv'
            ext = '.csv'
        elif ((ext != '.csv') and (ext != '.CSV')):
            print "\nError, %s CSV output file does not have .csv or .CSV extension\n" % opts.csv_output
            parser.print_help()
            sys.exit(1)

    election = Election(nseats=opts.nseats,
                        max_score=opts.max_score,
                        csv_input=csv_input,
                        csv_output=csv_output,
                        qtype=opts.quota_type)

    election.run_election(verbose=opts.verbose,
                          terse=opts.terse,
                          debug=opts.debug)
