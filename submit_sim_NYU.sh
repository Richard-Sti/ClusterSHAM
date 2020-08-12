# Lower bound only
string1="python3 simulation.py --name NYUmatch --type LF --cut 0"
string2="python3 simulation.py --name NYUmatch --type LF --cut 1"
string3="python3 simulation.py --name NYUmatch --type LF --cut 2"
string4="python3 simulation.py --name NYUmatch --type LF --cut 3"

$string1 &
$string2 &
$string3 &
$string4 &

# Lower and upper bound
string1="python3 simulation.py --name NYUmatch --type LF --cut 4"
string2="python3 simulation.py --name NYUmatch --type LF --cut 5"
string3="python3 simulation.py --name NYUmatch --type LF --cut 6"

$string1 &
$string2 &
$string3 &

# Lower bound only
string1="python3 simulation.py --name NYUmatch --type MF --cut 0"
string2="python3 simulation.py --name NYUmatch --type MF --cut 1"
string3="python3 simulation.py --name NYUmatch --type MF --cut 2"
string4="python3 simulation.py --name NYUmatch --type MF --cut 3"

$string1 &
$string2 &
$string3 &
$string4 &

# Lower and upper bound
string1="python3 simulation.py --name NYUmatch --type MF --cut 4"
string2="python3 simulation.py --name NYUmatch --type MF --cut 5"
string3="python3 simulation.py --name NYUmatch --type MF --cut 6"

$string1 &
$string2 &
$string3 
