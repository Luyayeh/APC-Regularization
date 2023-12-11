import re
import kaldiark
import numpy

def load_mean_var(mean_var_path):
    f = open(mean_var_path)
    line = f.readline()
    sum = numpy.array([float(e) for e in line[1:-2].split(', ')])
    line = f.readline()
    sum2 = numpy.array([float(e) for e in line[1:-2].split(', ')])
    nsamples = int(f.readline())

    feat_mean = sum / nsamples
    feat_var = sum2 / nsamples - feat_mean * feat_mean

    return feat_mean, feat_var


def parse_scp_line(line):
    m = re.match(r'(.+) (.+):(.+)', line)
    key = m.group(1)
    file = m.group(2)
    shift = int(m.group(3))

    return key, file, shift


def load_labels(file, shift):
    f = open(file)
    f.seek(shift)
    labels = f.readline().split()
    f.close()

    return labels

def add_awgn_noise(feat, snr):
    # Calculate noise power based on desired SNR
    signal_power = numpy.mean(feat ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    
    # Generate Gaussian-distributed noise with the same shape as the feature
    noise = numpy.random.randn(*feat.shape)
    
    # Scale the noise to the desired power
    scaled_noise = numpy.sqrt(noise_power) * noise
    
    # Add the noise to the feature
    noisy_feat = feat + scaled_noise
    
    return noisy_feat

def load_feat(file, shift, feat_mean=None, feat_var=None, snr=None):
    f = open(file, 'rb')
    f.seek(shift)
    feat = kaldiark.parse_feat_matrix(f)
    f.close()

    if feat_mean is not None and feat_var is not None:
        feat = (feat - feat_mean) / numpy.sqrt(feat_var)

    # Desired SNR for adding white noise
    if snr is not None:
        feat = add_awgn_noise(feat, snr)

    return feat


class LibriSpeech:
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None, snr=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var
        self.snr = snr

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __iter__(self):
        for i in self.indices:
            key, file, shift = self.feat_entries[i]
            feat = load_feat(file, shift, self.feat_mean, self.feat_var, self.snr)

            yield key, feat

