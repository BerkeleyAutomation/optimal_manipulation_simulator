import sys
import re


_, infile, outfile = sys.argv
logfile = open(infile, 'r')

lines = list(logfile)
logfile.close()

fails = []
for index in range(0, len(lines)):
	if lines[index].startswith("Fail"):
		fails.append(index)
		fails.append(index - 1)
		fails.append(index - 2)

newlines = [i for j, i in enumerate(lines) if j not in fails]

data = []
for line in newlines:
	if line.startswith("Terminated after"):
		data.append(re.search(r' (\d+.?\d*) ', line).group(1) + ", ")
	elif line.startswith("Optimization actually took"):
		data.append(re.search(r' (\d+.?\d*) ', line).group(1) + ", ")
	elif line.startswith("Run"):
		data.append("\n")
print data

logfile = open(outfile, 'a')
for line in data:
	logfile.write(line)

logfile.flush()