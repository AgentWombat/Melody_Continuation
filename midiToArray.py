import numpy as np
import re
from mido import MidiFile

# Midi files must: be at 120 bpm; be fully quantized to the eighth note (even lengths);
# only have one or no notes playing at a single time; have no notes carry over between measures;
# have no space between measures.


# given a Mido track, starting measure, and the amount of no-not beats 
# at the start of the program (zeros), returns a two-dimensional array
# with each index containing a measure.
# Starting at measure 7 should get all note messages in midi files generated
# by Logic Pro X.
def getMeasures(track, messageNumber = 7):

	maxMessageNumber = int(re.search(r'\d+', str(track)).group())


	measures = []

	#find initial leading zeros
	_, leadingZeros = getNote(track[messageNumber])
	index = 1
	while messageNumber + 1< maxMessageNumber: #add to measures array until the end of midi file.

		measure, messageNumber, leadingZeros = getMeasure(track, messageNumber, leadingZeros)


		measures.append(measure)

	return measures



# given a track, starting message number, and the number of beats
# before the first note, returns an array representing one measure
# given a track, starting message number, and the number of beats
# before the first note, returns an array representing one measure
def getMeasure(track, messageNumber, leadingZeros):
	measure = []

	currentBeat = 0

	#add leading zeros
	for i in range(leadingZeros):
		measure.append(0)
		currentBeat += 1

	# construct remainder of measure
	while currentBeat < 8:

		note, _ = getNote(track[messageNumber])

		measure.append(note)

		currentBeat += 1

		_, noteLength = getNote(track[messageNumber + 1])

		#add ones to represent how long a note is held
		for i in range(noteLength - 1):
			measure.append(1)
			currentBeat += 1

		# advance to "note_off" message

		_, blankSpaceLength = getNote(track[messageNumber + 2])

		for i in range(blankSpaceLength):
			measure.append(0)
			currentBeat += 1

		messageNumber += 2 # prepare for next while-loop iteration; advane to next "note_on" message


	leftOverBeats = len(measure) - 8 #This is how many beats beyond 8 are within the measure
	#it is also used to determine the number of leading zeros in the next measure


	for i in range(leftOverBeats): #remove excess zeros at the end of measure array
		measure.pop(-1)

	return measure, messageNumber, leftOverBeats

# given a valid 'note_on' or 'note_off' message, returns the note and the
# distance, in beats, from the previous message.
# if 'end_of_track' message is passed as argument, -1 is returned as note
# and the beats between the last note and the 'end_of_note' message
def getNote(message):# every other note will be an 'off' message

	if "end_of_track" in (str(message)):
		return -1, int(message.time/240)

	return int(message.note), int(message.time/240)





#converts a note to a one hot vector
#legacy
def note_to_onehot(note):
	converter = {
	60: 0,
	62: 1,
	64: 2,
	65: 3,
	67: 4,
	69: 5,
	71: 6,
	72: 7,
	1: 8,
	0: 9
	}

	vector = np.zeros(10, dtype = int)

	vector[converter[note[0]]] = 1

	return np.array(vector)


# Takes a two dimensional array (an array of measures) and converts it to a three dimensional array (an array of measures with one hot vectors as notes)
# If add_beat_vector, each note will also have an extra beats_per_measure dimensions to indicate on which beat the note is.
# If add_beat, an integer representing the current beat is added to the end of the vector
def to_onehot(m, add_beat_vector=False, add_beat = False):

	if (isinstance(m, np.ndarray) == False):
		m = np.array(m)

	converter = {
	60: 0,
	62: 1,
	64: 2,
	65: 3,
	67: 4,
	69: 5,
	71: 6,
	72: 7,
	1: 8,
	0: 9
	}

	dim = len(converter)

	if add_beat_vector:
		dim += len(m[0])
	if add_beat:
		dim += 1

	m2 = np.zeros((m.shape[0],m.shape[1], dim))

	for i, measure in enumerate(m):
		for j, note in enumerate(measure):
			m2[i][j][converter[note]] = 1

			if add_beat_vector:
				m2[i][j][len(converter) + j] = 1

			if add_beat:
				m2[i][j][-1] = j

	return m2

# Assumes an even number of measures
# If eight_to_eight (default), returns input and output arrays. Even indexed arrays become input, odd indexed arrays become output.
# Else, it parses inputs with memory_steps notes and single note outputs
def split_input_output(m, eight_to_eight = True, memory_steps = 8):

	if (isinstance(m, np.ndarray) == False):
		m = np.array(m)


	if (eight_to_eight):

		inputs = np.array([])

		outputs = np.array([])

		for i, measure in enumerate(m):

			if i%2 == 0:
				inputs = np.append(inputs, measure)

			else:
				outputs = np.append(outputs, measure)


		inputs = inputs.reshape((int(m.shape[0]/2), m.shape[1]))

		outputs = outputs.reshape((int(m.shape[0]/2), m.shape[1]))

		return inputs, outputs

	else:

		inputs = np.array([])

		outputs = np.array([])

		two_measure_pairs = m.reshape((int(m.shape[0]/2),m.shape[1]*2))

		for tmp in two_measure_pairs:
			for i in range(memory_steps, len(tmp)):
				inputs = np.append(inputs, tmp[i-memory_steps : i])

		outputs = np.array([tmp[memory_steps:] for tmp in two_measure_pairs])

		return inputs.reshape((outputs.size, 8)), outputs.flatten()



# Returns inputs and ouputs to be fed into a ML model from a specified midi_file
# If eight_to_eight (default), returns input and output arrays. Even indexed measures become input, odd indexed measures become output.
# Else, it parses inputs with memory_steps notes and single note outputs.
# Example with eight_to_eight = False and memory_steps = 2: [60,62,64,65,67,69] -> [[60,62],[62,64], [64,65], [65,67]] , [64,65,67,69]
def load_data(midi_file, eight_to_eight = True, memory_steps = 8, onehot = True, add_beat_vector=False, add_beat = False):

	mid = MidiFile(midi_file)

	measures = getMeasures(mid.tracks[0], 7)


	inputs, outputs = split_input_output(measures, eight_to_eight, memory_steps)

	if onehot:
		inputs = to_onehot(inputs, add_beat_vector, add_beat)

		outputs = to_onehot([[output] for output in outputs])


	return inputs, outputs