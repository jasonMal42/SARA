# PROGRAM HEADER, Similarity Assessment of Raman Arrays
"""
Filename: SARA.py
Author: Jason Malenfant
Date created: 2022-02-22
License: CC BY-NC-SA 4.0
"""

import multiprocessing
import os
import numpy as np

from struct import unpack
from scipy import special


def matching_spectra_files(directory, extension: str = ".txt"):  # Relies on user's file naming and OS sorting

    filesList = list()
    allBaseNames = list()
    for file in os.listdir(directory):
        if file.endswith(extension):
            filesList.append(os.path.join(directory, file))
            allBaseNames.append(os.path.splitext(file)[0])

    if len(filesList) % 2 != 0:
        print("ERROR! Files list has an odd number of items.")
        raise ValueError

    baseNamesList = [(allBaseNames[i], allBaseNames[i + 1]) for i in range(0, len(allBaseNames) - 1, 2)]

    return list(zip(filesList[0::2], filesList[1::2])), baseNamesList


def all_spectra_files(directory, exp_extension: str = ".txt", theo_extension: str = ".stk"):  # Refactor?
    filePathsExp, baseNamesExp, filePathsTheo, baseNamesTheo = [], [], [], []
    filePathsUnknown, baseNamesUnknown = [], []
    assignedFilePaths, assignedBaseNames = [], []

    # Scrub directory and try to identify the types of spectra
    if exp_extension == theo_extension:
        print("File extensions are the same, assigning spectrum types based on filenames...")
        for file in os.listdir(directory):
            if file.endswith(exp_extension):
                if 'exp' in file.lower():
                    filePathsExp.append(os.path.join(directory, file))
                    baseNamesExp.append(os.path.splitext(file)[0])
                elif ('calc' in file.lower()) or ('theo' in file.lower()) or ('dft' in file.lower()):
                    filePathsTheo.append(os.path.join(directory, file))
                    baseNamesTheo.append(os.path.splitext(file)[0])
                else:
                    filePathsUnknown.append(os.path.join(directory, file))
                    baseNamesUnknown.append(os.path.splitext(file)[0])
                    print("Spectra " + str(file) + " is of unknown nature!")
    else:
        print("Assigning spectrum types based on file extensions...")
        for file in os.listdir(directory):
            if file.endswith(exp_extension):
                filePathsExp.append(os.path.join(directory, file))
                baseNamesExp.append(os.path.splitext(file)[0])
            elif file.endswith(theo_extension):
                filePathsTheo.append(os.path.join(directory, file))
                baseNamesTheo.append(os.path.splitext(file)[0])

    # Sanity checks
    if len(baseNamesExp) == 0 and len(baseNamesTheo) == 0 and len(baseNamesUnknown) == 0:
        raise ValueError("ERROR: No spectra were parsed!")
    if len(baseNamesExp) > 0 and len(baseNamesTheo) > 0 and len(baseNamesUnknown) > 0:
        if exp_extension == theo_extension:
            raise ValueError("ERROR: Spectra types are ambiguous!")

    # Assign spectra pairs and create tuples
    if len(filePathsUnknown) == 0:  # Best case scenario
        for expIndex, experimental in enumerate(filePathsExp):
            for theoIndex, theoretical in enumerate(filePathsTheo):
                assignedFilePaths.append((experimental, theoretical))
                assignedBaseNames.append((baseNamesExp[expIndex], baseNamesTheo[theoIndex]))

    elif len(filePathsExp) > 0 and len(filePathsTheo) == 0:
        for expIndex, experimental in enumerate(filePathsExp):
            for unknownIndex, unknown in enumerate(filePathsUnknown):
                assignedFilePaths.append((experimental, unknown))
                assignedBaseNames.append((baseNamesExp[expIndex], baseNamesUnknown[unknownIndex]))

    elif len(filePathsExp) == 0 and len(filePathsTheo) > 0:
        for unknownIndex, unknown in enumerate(filePathsUnknown):
            for theoIndex, theoretical in enumerate(filePathsTheo):
                assignedFilePaths.append((unknown, theoretical))
                assignedBaseNames.append((baseNamesUnknown[unknownIndex], baseNamesTheo[theoIndex]))

    else:
        for unknownIndex1, unknown1 in enumerate(filePathsUnknown):
            for unknownIndex2, unknown2 in enumerate(filePathsUnknown):
                if unknown1 != unknown2:
                    assignedFilePaths.append((unknown1, unknown2))
                    assignedBaseNames.append((baseNamesUnknown[unknownIndex1], baseNamesUnknown[unknownIndex2]))

    return assignedFilePaths, assignedBaseNames


def normalize(intensities):
    minimum, maximum = intensities.min(), intensities.max()
    intensities = (intensities - minimum) / (maximum - minimum)  # This uses the min-max algorithm.
    return intensities


def gen_from_peaks(data, half_width: int = 60):
    wave_max = int(round(data[:, 0].max() + 200))
    wave_min = max(0, int(round(data[:, 0].min() - 200)))
    nbPoints = wave_max - wave_min + 1

    spectrum = np.empty((nbPoints, 2))
    spectrum[:, 0] = np.linspace(wave_min, wave_max, nbPoints)
    spectrum[:, 1] = np.zeros(nbPoints)

    for i, peak in enumerate(data):
        mu, inputIntensity = int(round(peak[0])), peak[1]
        beginWave, endWave = max(0, (mu - half_width)), max(0, (mu + half_width))

        nbTicks = len(range(beginWave, (endWave + 1), 1))
        curvePoints = np.linspace((beginWave - mu), (endWave - mu), nbTicks)

        voigtFactors = special.voigt_profile(curvePoints, 1, 4)

        outputIntensities = inputIntensity * normalize(voigtFactors)

        padStart, padEnd = (beginWave - wave_min), (wave_max - endWave)
        intensitiesPadded = np.pad(outputIntensities, (padStart, padEnd), 'constant', constant_values=(0, 0))
        signalMatrix = np.zeros((nbPoints, 2))
        signalMatrix[:, 1] = intensitiesPadded

        spectrum = spectrum + signalMatrix  # The wavenumbers were set, so we're adding the intensities on top.

    return spectrum


def gen_from_wdf(full_path, verbose: bool = False):
    def parse_block_metadata():
        block_title = f.read(4).decode("ascii")
        f.seek(4, 1)
        block_length = unpack("<Q", f.read(8))[0]
        return block_title, block_length

    class RenishawBlock:
        def __init__(self, position):
            f.seek(position, 0)
            self.block_type, self.block_size = parse_block_metadata()
            self.beginOffset = position
            self.intensity, self.wavenumbers = None, None
            self.y_list_is_present, self.y_list_is_empty = False, False
            self.found_nb_points = 0
            self.nbAcc, self.nbSpectra, self.laserWavenumber = None, None, None
            
        def __repr__(self):
            return f'Block type {self.block_type}'
            
        

        def read_wdf1(self):  # WDF1
            f.seek(44, 1)
            readNbDataPoints = unpack("<I", f.read(4))[0]
            runTarget = unpack("<Q", f.read(8))[0]
            runCount = unpack("<Q", f.read(8))[0]
            if runTarget != runCount:
                if verbose:
                    print("Warning: the run wasn't complete, and the spectra could need resizing.")
            self.nbAcc = unpack("<I", f.read(4))[0]
            self.nbSpectra = unpack("<I", f.read(4))[0]
            nbWavenumbers = unpack("<I", f.read(4))[0]
            if nbWavenumbers != readNbDataPoints:
                raise ValueError(
                    "The number of expected data points is different than the amount of wavenumber points obtained.")
            self.found_nb_points = readNbDataPoints
            f.seek(64, 1)
            self.laserWavenumber = unpack("<f", f.read(4))[0]

        def _read_data(self, nb_data_points):  # DATA
            intensity_data = np.fromfile(f, dtype="float32", count=nb_data_points)
            self.intensity = intensity_data

        def _read_ylst(self):  # YLST
            self.y_list_is_present = True
            if self.block_size <= 28:
                self.y_list_is_empty = True

        def _read_xlst(self, nb_data_points):  # XLST
            # Read
            wireDataType = unpack("<I", f.read(4))[0]  # We want a value of 1 which means 'Frequency'
            if wireDataType not in [0, 1, 19]:
                raise TypeError("This data type cannot be used as the photon energy axis.")
            elif wireDataType == 0:
                if verbose:
                    print("Attention: X axis is of the potentially incorrect 'arbitrary' data type.")
            elif wireDataType == 19:
                if verbose:
                    print("Attention: X axis is of the possibly incorrect 'spectral' data type.")

            # Get units
            wavenumberUnitType = unpack("<I", f.read(4))[0]  # Value of 1 means cm^-1
            if wavenumberUnitType not in range(6):
                raise ValueError("Incorrect units for wavenumber axis.")
            elif wavenumberUnitType == 0:
                raise ValueError("Arbitrary units cannot be converted to reciprocal centimeters.")

            # Convert data to reciprocal centimeters
            wavenumber_data = np.fromfile(f, dtype="float32", count=nb_data_points)
            if wavenumberUnitType in [2, 3]:  # nm
                wavenumber_data = 1E7 / wavenumber_data
            elif wavenumberUnitType == 4:  # eV
                wavenumber_data = 1.23981E-4 * wavenumber_data
            elif wavenumberUnitType == 5:  # µm
                wavenumber_data = 1E4 / wavenumber_data

            self.wavenumbers = wavenumber_data

        def read_block(self, block, nb_data_points):
            block_name = block.lower()
            method_to_execute = f"_read_{block_name}"
            if hasattr(self, method_to_execute) and callable(func := getattr(self, method_to_execute)):
                func(nb_data_points)

    with open(full_path, 'rb') as f:

        f.seek(0)
        wdfBlock = RenishawBlock(0)
        wdfBlock.read_wdf1()

        if wdfBlock.block_type != "WDF1":
            raise ValueError("File does not start with correct header block!")
        cumulativePosition = wdfBlock.block_size
        specDataPoints = wdfBlock.found_nb_points

        if verbose:
            print(f"")

        wdf_spectrum = np.empty((specDataPoints, 2))

        body_blocks = []
        for n_block in range(4):
            body_blocks.append(RenishawBlock(cumulativePosition))
            current_block_type = body_blocks[n_block].block_type

            if verbose:
                print(f"Block {n_block} at byte {cumulativePosition} is type {current_block_type}.")

            cumulativePosition += body_blocks[n_block].block_size

            if current_block_type == "XLST":
                body_blocks[n_block].read_block("XLST", specDataPoints)
                wdf_spectrum[:, 0] = body_blocks[n_block].wavenumbers
            elif current_block_type == "DATA":
                body_blocks[n_block].read_block("DATA", specDataPoints)
                wdf_spectrum[:, 1] = body_blocks[n_block].intensity

    if (np.count_nonzero(wdf_spectrum[:, 0]) <= 0) or (np.count_nonzero(wdf_spectrum[:, 1]) <= 0):
        raise ValueError("There is a missing axis to the spectrum!")

    if verbose:
        print("WDF-Extracted spectrum: ")
        print(wdf_spectrum)

    return wdf_spectrum


def fetch_experimental_spectrum(file_path):
    if file_path.endswith('.txt'):  # First let's try an excel-styled csv
        with open(file_path, 'r') as f:
            if '#Wave		#Intensity' in f.read():  # which is the header of a Renishaw WiRE-type file
                readArray = np.genfromtxt(file_path, dtype='float32')

            else:  # Try generic Excel-based csv
                readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                          filling_values=0, missing_values=0)

    elif file_path.endswith('.wdf'):  # This is the Renishaw WiRE proprietary format.
        readArray = gen_from_wdf(file_path, False)

    else:  # Also try generic Excel-based csv if file doesn't have a '.txt' extension.
        readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                  filling_values=0, missing_values=0)

    if len(readArray.shape) != 2:
        raise ValueError("Parsed data does not have the expected two axis.")

    return readArray


def fetch_theoretical_spectrum(file_path):
    if file_path.endswith('.dat'):  # would indicate an ORCA-rendered spectrum
        readArray = np.genfromtxt(file_path, dtype='float32')

    elif file_path.endswith('.stk'):  # would indicate ORCA peak data
        peakInformation = np.genfromtxt(file_path, dtype='float32')
        readArray = gen_from_peaks(peakInformation)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            if '# Raman Activity Spectrum' in f.read():  # header of a Gaussian-type file
                readArray = np.genfromtxt(file_path, comments='#', usecols=(0, 1))

            elif '#Wave		#Intensity' in f.read():  # which is the header of a Renishaw WiRE-type file
                readArray = np.genfromtxt(file_path, dtype='float32')

            else:  # Try generic Excel-based csv
                readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                          filling_values=0, missing_values=0)

    else:  # last resort, try assuming Excel-based csv in case of undefined extension.
        readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                  filling_values=0, missing_values=0)

    if len(readArray.shape) != 2:
        raise ValueError("Parsed data does not have the expected two axis.")

    return readArray


def fetch_any_spectrum(file_path):
    if file_path.endswith('.dat'):  # would indicate an ORCA-rendered spectrum
        readArray = np.genfromtxt(file_path, dtype='float32')

    elif file_path.endswith('.stk'):  # would indicate ORCA peak data
        peakInformation = np.genfromtxt(file_path, dtype='float32')

        print("Generating actual spectrum from STK peak data...")
        readArray = gen_from_peaks(peakInformation)

    elif file_path.endswith('.wdf'):  # This is the Renishaw WiRE proprietary format.
        readArray = gen_from_wdf(file_path, False)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            if '# Raman Activity Spectrum' in f.read():  # header of a Gaussian-type file
                readArray = np.genfromtxt(file_path, comments='#', usecols=(0, 1))

            else:
                readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                          filling_values=0, missing_values=0)

    else:  # last resort, try assuming Excel-based csv.
        readArray = np.genfromtxt(file_path, dtype='float32', delimiter=',', comments='#', autostrip=True,
                                  filling_values=0, missing_values=0)

    if len(readArray.shape) != 2:
        raise ValueError("Parsed data does not have the expected two axis.")

    return readArray


def interpolate_spec(spectrum, resolution=1):
    inputWavenumbers = spectrum[:, 0]
    rangeMin = int(np.rint(inputWavenumbers.min()))
    rangeMax = int(np.rint(inputWavenumbers.max()))
    outputWavenumbers = range(rangeMin, (rangeMax + 1), resolution)

    outputIntensities = np.interp(outputWavenumbers, inputWavenumbers, spectrum[:, 1])
    interpolated = np.empty((len(outputWavenumbers), 2))
    interpolated[:, 0], interpolated[:, 1] = outputWavenumbers, outputIntensities

    return interpolated


def truncate_specs(spec_1, spec_2):
    lowSlice = max(spec_1[:, 0].min(), spec_2[:, 0].min(), 200)
    highSlice = min(spec_1[:, 0].max(), spec_2[:, 0].max(), 2800)

    mask_1 = (spec_1[:, 0] <= highSlice) & (spec_1[:, 0] >= lowSlice)
    mask_2 = (spec_2[:, 0] <= highSlice) & (spec_2[:, 0] >= lowSlice)

    truncated_1 = np.empty((np.count_nonzero(mask_1), 2))
    truncated_1[:, 0] = np.extract(mask_1, spec_1[:, 0])
    truncated_1[:, 1] = np.extract(mask_1, spec_1[:, 1])

    truncated_2 = np.empty((np.count_nonzero(mask_2), 2))
    truncated_2[:, 0] = np.extract(mask_2, spec_2[:, 0])
    truncated_2[:, 1] = np.extract(mask_2, spec_2[:, 1])

    return truncated_1, truncated_2


def circular_arc(k, cumulative_previous_k, direction: str):
    o = cumulative_previous_k  # Serves as an offset
    genRange = (cumulative_previous_k, cumulative_previous_k + k)

    x = np.linspace((genRange[0] + 0.000001), (genRange[1] - 0.000499), 2000)

    if direction == "up":
        y = k * (1 - ((((k ** 2 - (x - o) ** 2) ** 0.5).real - o) / k))
    else:
        y = -k * (1 - (((k ** 2 - (x - k - o) ** 2) ** 0.5).real / k)) + k + o

    finalArc = np.empty((2000, 2))
    finalArc[:, 0], finalArc[:, 1] = x, y

    return finalArc


def compress_spectrum(input_intensities, flavor="harsh"):
    if flavor == "soft":
        arcData = [{"k": 1, "type": "down"}, {"k": 2, "type": "up"}, {"k": 2, "type": "down"},
                   {"k": 4, "type": "up"}, {"k": 4, "type": "down"}, {"k": 8, "type": "up"},
                   {"k": 8, "type": "down"}, {"k": 16, "type": "up"}, {"k": 16, "type": "down"}]
    elif flavor == "harsh":
        arcData = [{"k": 1, "type": "down"}, {"k": 2, "type": "down"}, {"k": 4, "type": "down"},
                   {"k": 8, "type": "down"}]
    elif flavor == "harsh2":
        arcData = [{"k": 1, "type": "up"}, {"k": 2, "type": "up"}, {"k": 4, "type": "up"},
                   {"k": 8, "type": "up"}]
    elif flavor == "brutal":
        arcData = [{"k": 1, "type": "down"}, {"k": 2, "type": "down"}, {"k": 4, "type": "down"}]
    elif flavor == "grey":
        arcData = [{"k": 1, "type": "down"}, {"k": 1, "type": "up"}]
    else:
        arcData = []

    if flavor == "discrete":
        compressionFunction = np.empty((0, 2))
        compressionFunction = np.append(compressionFunction,  # This discrete flavor is made of flat steps defined as:
                                        [[0, 0], [0.005, 0], [0.0050001, 0.015], [0.01, 0.015], [0.0100001, 0.04],
                                         [0.17, 0.04], [0.1700001, 0.25], [0.4, 0.25], [0.4000001, 0.62], [0.8, 0.62],
                                         [0.8000001, 0.95], [0.97, 0.95], [0.9700001, 1], [1, 1]], axis=0)

    elif flavor == "jay-cutoff":
        x = np.linspace(0, 1, 2000)
        # y = 1 - (1.16 * ((5*x+1.1) ** (-0.7 * (x+1)))) + ((46 ** (0 -( ( (x+0.014)**2 ) / (2 * (0.0378**2)) ))) * 0.11)
        y = 1 - (1.16 * ((5*x+1.1) ** (-0.7 * (x+1)))) + (0.11*(46 ** ( -350*x**2 -9.8*x -0.0686 ) ) )
        compressionFunction = np.empty((2000, 2))
        compressionFunction[:, 0], compressionFunction[:, 1] = normalize(x), normalize(y)

    elif flavor == "mat-cutoff":
        x = np.linspace(0, 0.03, 500)
        y = 0.03 - (0.03 * np.cos((x/0.03*(np.pi/2))))
        x2 = np.linspace(0.0301, 1, 2500)
        y2 = 0.03 + np.sqrt(x2-0.03)
        x, y = np.append(x, x2), np.append(y, y2)
        compressionFunction = np.empty((3000, 2))
        compressionFunction[:, 0], compressionFunction[:, 1] = normalize(x), normalize(y)
        
    elif flavor == "log-medium":
        x = np.linspace(0, 1, 2000)
        y = (0.88 * (np.log10(x + 0.08))) + 0.97
        compressionFunction = np.empty((2000, 2))
        compressionFunction[:, 0], compressionFunction[:, 1] = normalize(x), normalize(y)

    else:
        compressionFunction = np.empty((0, 2))
        totalK = 0
        for definition in arcData:  # Function is then generated from circle quarters...
            currentArc = circular_arc(definition["k"], totalK, definition["type"])
            compressionFunction = np.append(compressionFunction, currentArc, axis=0)
            totalK += definition["k"]

        compressionFunction[:, 0] = normalize(compressionFunction[:, 0])
        compressionFunction[:, 1] = normalize(compressionFunction[:, 1])

    output_intensities = np.interp(input_intensities, compressionFunction[:, 0], compressionFunction[:, 1])

    return output_intensities


def pre_process_single_spec(spectrum, rez: int):  # Simpler dev function for a single spectrum only.
    sorted_spec = spectrum[np.argsort(spectrum[:, 0])]
    scaled_spec = interpolate_spec(sorted_spec, rez)
    norm_spec = scaled_spec.copy()
    norm_spec[:, 1] = normalize(scaled_spec[:, 1])

    return norm_spec


def pre_process_specs(experimental_spec, theoretical_spec, theory_scale_factor, compression="soft"):
    sorted_exp = experimental_spec[np.argsort(experimental_spec[:, 0])]
    sorted_theo = theoretical_spec[np.argsort(theoretical_spec[:, 0])]

    sorted_theo[:, 0] = sorted_theo[:, 0] * theory_scale_factor

    resampled_exp, resampled_theo = interpolate_spec(sorted_exp), interpolate_spec(sorted_theo)
    truncated_exp, truncated_theo = truncate_specs(resampled_exp, resampled_theo)

    normalized_exp, normalized_theo = truncated_exp.copy(), truncated_theo.copy()
    normalized_exp[:, 1] = normalize(truncated_exp[:, 1])
    normalized_theo[:, 1] = normalize(truncated_theo[:, 1])

    if compression.lower() == "none":
        return normalized_exp, normalized_theo

    compressed_exp, compressed_theo = normalized_exp.copy(), normalized_theo.copy()
    compressed_exp[:, 1] = compress_spectrum(compressed_exp[:, 1], compression)
    compressed_theo[:, 1] = compress_spectrum(compressed_theo[:, 1], compression)

    return compressed_exp, compressed_theo


def auto_correlation(spectrum, k: int = 100):
    correlationData = np.empty((0, 2))

    for r in range(-k, (k + 1), 1):
        sumOfProducts = 0.0

        for i in range(0, spectrum.shape[0], 1):
            if 0 < (i + r) < spectrum.shape[0]:
                sumOfProducts = sumOfProducts + (abs(spectrum[i, 1]) * abs(spectrum[(i + r), 1]))

        correlationData = np.append(correlationData, [[r, sumOfProducts]], axis=0)

    return correlationData


def cross_correlation(spec_1, spec_2, k: int = 100):
    if spec_1.shape != spec_2.shape:
        raise ValueError("FATAL ERROR! Spectra do not have the same shape, aborting...")

    correlationData = np.empty((0, 2))

    for r in range(-k, (k + 1), 1):
        sumOfProducts = 0.0

        for i in range(0, (spec_1.shape[0]), 1):
            if 0 < (i + r) < spec_1.shape[0]:
                sumOfProducts = sumOfProducts + (abs(spec_1[i, 1]) * abs(spec_2[(i + r), 1]))

        correlationData = np.append(correlationData, [[r, sumOfProducts]], axis=0)

    return correlationData


def integrate_correlation(criterion_shape, correlation_data, alpha, beta: float = 4):  # Default Beta == 4
    if criterion_shape == "karfunkel":
        weightsInsideRange = np.fromiter(((2 / (2 + (alpha * (abs(r) ** beta)))) for r in range(-65, 66, 1)), float)
        weightsInsideRange = normalize(weightsInsideRange)
    elif criterion_shape == "triangle":
        weightsInsideRange = np.interp(range(-alpha, (alpha + 1), 1), [-alpha, 0, alpha], [0, 1, 0])
    else:
        raise ValueError("Unknown criterion function.")

    if weightsInsideRange.shape[0] < correlation_data.shape[0]:
        pointsToAdd = round((correlation_data.shape[0] - weightsInsideRange.shape[0]) / 2)
        weights = np.pad(weightsInsideRange, (pointsToAdd, pointsToAdd), mode='constant', constant_values=0)
    elif weightsInsideRange.shape[0] > correlation_data.shape[0]:
        raise ValueError('Amount of correlation points is smaller than width of requested weight function.')
    else:
        weights = weightsInsideRange

    if weights.shape[0] == correlation_data.shape[0]:
        weightedCorrelation = weights * correlation_data[:, 1]
    else:
        print("ERROR while preparing integration weights.")
        weightedCorrelation = np.empty((0, 1))

    integral = np.sum(weightedCorrelation)

    return integral


def sara_core(path_exp, path_theo, harmonic_correction, width, compression="none", identifier=("null", "null")):

    experimentalSpec = fetch_experimental_spectrum(path_exp)
    theoreticalSpec = fetch_theoretical_spectrum(path_theo)

    processedExp, processedTheo = pre_process_specs(experimentalSpec, theoreticalSpec, harmonic_correction, compression)

    autoPatternExp, autoPatternTheo = auto_correlation(processedExp), auto_correlation(processedTheo)
    crossPattern = cross_correlation(processedExp, processedTheo)

    autoCorrelIntegralExp = integrate_correlation("karfunkel", autoPatternExp, width)
    autoCorrelIntegralTheo = integrate_correlation("karfunkel", autoPatternTheo, width)
    crossCorrelIntegral = integrate_correlation("karfunkel", crossPattern, width)

    score = round(((crossCorrelIntegral / ((autoCorrelIntegralExp * autoCorrelIntegralTheo) ** 0.5).real) * 100), 2)

    return score, identifier[0], identifier[1]


def create_csv_1d(scores, axis):
    separators = {"horizontal": ",", "vertical": '\n'}
    if axis.lower() not in separators:
        raise ValueError("Unknown axis direction for preparing 1D CSV string.")

    csvString = ""
    print("Ready to export " + str(len(scores)) + " scores as a ribbon of numbers:")
    for scoreIndex, score in enumerate(scores):
        csvString += str(score[0]) + ',' + str(score[1]) + '+' + str(score[2]) + separators[axis.lower()]


def create_csv_2d(scores):
    allExpFiles, allTheoFiles = [], []
    for score in scores:
        allExpFiles.append(score[1])
        allTheoFiles.append(score[2])
    expFilesSet, theoFilesSet = list(set(allExpFiles)), list(set(allTheoFiles))
    dataShape = (len(expFilesSet), len(theoFilesSet))
    csvString = ""

    if (dataShape[0] * dataShape[1]) > len(scores):
        print("WARN: there may not be enough scores to fill the matrix.")

    print("Ready to export " + str(dataShape[0]) + "x" + str(dataShape[0]) + " score matrix:")

    headerString = '#----,'
    for theoFile in theoFilesSet:
        headerString += (str(theoFile) + ",")
    csvString += (headerString[0:-1] + "\n")

    for expFile in expFilesSet:
        currentLineString = (str(expFile) + ",")

        for theoFile in theoFilesSet:
            try:
                tupleIndex = 9999
                for i, guess in enumerate(scores):
                    if guess[1:3] == (expFile, theoFile):
                        tupleIndex = i
                scoreToWrite = scores[tupleIndex][0]
            except ValueError:
                scoreToWrite = "NaN"
            except IndexError:
                scoreToWrite = "NaN"

            currentLineString += (str(scoreToWrite) + ",")

        csvString += (currentLineString[0:-1] + "\n")

    return csvString


def write_to_file(string, export_path):
    with open(export_path, "w") as f:
        f.write(string)
    return


if __name__ == '__main__':
    mainPath = r"G:\My Drive\2021-22 Research\DFT-Raman\Dataset 1\compary"
    filesToParse, baseNames = all_spectra_files(mainPath, ".txt", ".txt")
    # filesToParse = [((mainPath + "\\J-exp-NiAc.txt"), (mainPath + "\\J-d-tight.out.raman.stk"))]
    # baseNames = [("J-exp-NiAc", "J-d-tight.out.raman")]

    if len(baseNames) == 0:
        raise FileNotFoundError("No pairs of files that pass all the criteria were parsed.")
    for pair in baseNames:
        if len(pair) != 2:
            raise ValueError("One of the tuples doesn't have the expected 2 filenames.")
        elif pair[0] == pair[1]:
            raise ValueError("There is a pair with a duplicate filename.")

    nbThreads = int(round(os.cpu_count() / 2))

    widthParam = 0.00001  # Width parameter for weighting function applied on correlation integrals, old default 0.00005 new default 0.00001
    corrFactor = 0.98  # Anharmonic correction factor, applied on all theoretical frequencies
    compressMethod = "jay-cutoff"

    print("Ready to calculate on " + str(nbThreads) + " threads with a correction factor of " + str(corrFactor) +
          " for the DFT spectrum, with a width parameter of " + str(widthParam) + ".")
    if compressMethod == "none":
        print("No compression is being applied in the intensities.")
    else:
        print("Using the " + compressMethod + " compression scheme on the intensity of both spectra!")
    print("")
    print("Calculating scores for " + str(len(filesToParse)) + " pairs of spectra...")
    print("")

    threadPool = multiprocessing.Pool(nbThreads)
    parallelCalcsToSubmit = []

    for pairIndex, pair in enumerate(filesToParse):
        parallelCalcsToSubmit.append(
            threadPool.apply_async(sara_core, args=(pair[0], pair[1], corrFactor, widthParam,
                                   compressMethod, baseNames[pairIndex]))
                                            )

    scoreResults = [jobExecute.get() for jobExecute in parallelCalcsToSubmit]  # Submit jobs

    print(create_csv_2d(scoreResults))  # TODO Logic to choose csv type automatically? (Make sure 1D's show spec names)
    # write_to_file(create_csv_2d(scoreResults), "/home/jay/Desktop/SARA-output.csv")

    exit(0)
