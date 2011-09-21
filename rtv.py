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

qtypes = ['droop',
          'hare',
          'droop-nseats',
          'droop-max-score',
          'hagenbach-bischoff']

# Set up for Hare or Droop quotas:
def calc_quota(n,
               nseats=DEFAULT_NSEATS,
               max_score=DEFAULT_MAX_SCORE,
               qtype='droop'):
    """\
    Return the quota based on qtype:

    'droop'              => Droop = int(Nvotes / (Nseats + 1)) + 1
    'hare'               => Hare  = Nvotes / Nseats
    'droop-nseats'       => Droop with Nvotes*Nseats votes,
                            then divide by Nseats.
    'droop-max-score'    => Droop with Nvotes*max_score votes,
                            then divide by max_score
    'hagenbach-bischoff' => Nvotes / (Nseats + 1)
    """

    fn = float(n)
    fs = float(nseats)
    fm = float(max_score)
    fsp1 = fs + 1.0

    # We implement a CASE switch construction using a dict:
    return {'droop':              (float(int(fn/fsp1)) + 1.0),
            'hare':               (fn/fs),
            'droop-nseats':       ((float(float(int(fn*fs/fsp1)) + 1.0))/fs),
            'droop-max-score':    ((float(float(int(fn*fm/fsp1)) + 1.0))/fm),
            'hagenbach-bischoff': (fn/fsp1)}[qtype]

class Ballot(dict):
    def __init__(self,csv_string='',cand_list=[],offset_score=0):
        # Parse the csv_string
        scores = []
        for i, v in enumerate(csv_string.rstrip().split(',')):
            if v:
                intv = int(v)
                if intv:
                    scores.append((cand_list[i],intv+offset_score))
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
                 nseats=DEFAULT_NSEATS,
                 offset_score=0):
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

        self.max_score = 0
        if ballots:
            self.ballots = ballots

            # Determine maximum score in existing ballots
            self.max_score = max(v
                                 for ballot in self.ballots
                                 for v in ballot.values())
            print "Max score found in existing ballots = %d" % self.max_score
        else:
            self.ballots = []

        self.candidates = candidates
        if csv_input:
            if csv_input == '-':
                self.csv_ballots(stdin=True,
                                 offset_score=offset_score)
            else:
                self.csv_ballots(filename=csv_input,
                                 offset_score=offset_score)

        # Maximum Range score:
        self.n_score = self.max_score + 1

        # Initialize lists and sets of candidates:

        self.seated = set([])
        self.ordered_seated = []
        self.standing = self.candidates
        self.ordered_candidates=sorted(self.candidates)


        if csv_output:
            if csv_output == '-':
                self.csv_output = sys.stdout
            else:
                self.csv_output = open(csv_output, 'w')

        # Count the number of votes
        self.nvotes = len(self.ballots)

        # Calculate quota
        self.quota = calc_quota(self.nvotes,
                                nseats=self.nseats,
                                max_score=self.max_score,
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
                    stdin=False,
                    offset_score=0):
        "Read ballots from a csv file.  First line is names of candidates."
        if stdin:
            f = sys.stdin
        else:
            f = open(filename,'r')

        # List of candidate names in the first line:
        keys = f.readline().rstrip().split(',')
        self.candidates.update(set(keys))

        max_score = 0

        # Following lines are the ballots:
	# A completely empty ballot could be construed as a vote for
	# "None of the Above".  Depending on how votes are counted, this could
	# force a run-off.
        for line in f:
            ballot = Ballot(line,keys,offset_score)
            max_score = max(max_score, max(ballot.values()))
	    self.ballots.append(ballot)

        print "Maximum score found in CSV input = %d" % max_score

        self.max_score = max(self.max_score,max_score)
        print "Maximum score after CSV ballots merged = %d\n" % self.max_score

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
                if n == 0:
                    ballot.rescale == 0.0 # Ignore this ballot next time
                else:
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

        # Extract winner of this round:
	(winner, win_score) = ordered_scores[0]
	csv_line = self.print_running_total(ordered_scores)

        # Check for tied winning scores:
	tied_scores = dict([(cand,score)
			    for cand, score in total.iteritems()
			    if score == win_score])

        if len(tied_scores) > 1:
            # (not doing anything at this point ...)
            print "\nUh-oh!  There is a tie!"
            print "Tied candidates:"
            for c, score in tied_scores.iteritems():
                print "\t%s: %g" % (c, score)

	# Find the winner's corresponding locksum:
	lockval = locksum[winner]

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

    parser.add_option('-q',
                      '--quota-type',
                      type='string',
                      default='droop',
                      help=fill(dedent("""\
                      Quota type used in election.

                      'droop' = Droop   = Nballots /(Nseats + 1) + 1,
                                          dropping fractional part.

                      'hare'  = Hare    = Number of ballots divided by number
                                          of seats.

                      'droop-nseats'    = Droop based on Nballots * Nseats
                                          votes, then divide by Nseats.
                                          Reduces to traditional Droop
                                          when Nseats is 1.

                      'droop-max-score' = Droop based on Nballots * max_score
                                          votes, then divide by max_score.

                      'hagenbach-bischoff' = Nballots / (Nseats + 1).
                                             Technically, this may allow
                                             exactly 50% of the ballots to
                                             select a majority of seats,
                                             or the left-out votes could
                                             meet quota for an extra seat.

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

    parser.add_option('-f',
                      '--offset-score',
                      type=int,
                      default=0,
                      help="Increase all non-zero input scores by OFFSET_SCORE.  [Default:  0]")

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
        print "Reading CSV input from stdin"
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

        print "Reading CSV input from", csv_input

    if (csv_output == "-"):
        print "Appending CSV output to stdout\n"
    else:

        ext = os.path.splitext(csv_output)[1]

        if not ext:
            csv_output += '.csv'
            ext = '.csv'
        elif ((ext != '.csv') and (ext != '.CSV')):
            print "\nError, %s CSV output file does not have .csv or .CSV extension\n" % opts.csv_output
            parser.print_help()
            sys.exit(1)

        print "Writing CSV output to", csv_output, "\n"

    election = Election(nseats=opts.nseats,
                        csv_input=csv_input,
                        csv_output=csv_output,
                        qtype=opts.quota_type,
                        offset_score=opts.offset_score)

    election.run_election(verbose=opts.verbose,
                          terse=opts.terse,
                          debug=opts.debug)
