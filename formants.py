import extract_features as features
import argparse
from helpers.textgrid import *
from helpers.utilities import *
from load_estimation_model import load_estimation_model


def predict_from_times(wav_filename, preds_filename, begin, end, csv_export=True):
    tmp_features_filename = generate_tmp_filename("txt")
    #tmp_features_filename = "temp/" + next(tempfile._get_candidate_names()) + ".txt"
    print("Input Array Path: " +  tmp_features_filename)

    predictions = None
    # if begin > 0.0 or end > 0.0:
    print(wav_filename + " interval " + str(begin) + "-" + str(end) + ":")
    features.create_features(wav_filename, tmp_features_filename, begin, end)
    predictions = load_estimation_model(tmp_features_filename, preds_filename, begin, end, csv_export=csv_export)
        #easy_call("luajit load_estimation_model.lua " + tmp_features_filename + ' ' + preds_filename)
    # else:
    #     features.create_features(wav_filename, tmp_features_filename)
    #     easy_call("luajit load_tracking_model.lua " + tmp_features_filename + ' ' + preds_filename)
    
    delete_temp_files()
    return predictions


def predict_from_textgrid(wav_filename, preds_filename, textgrid_filename, textgrid_tier):
    print(wav_filename)

    if os.path.exists(preds_filename):
        os.remove(preds_filename)

    textgrid = TextGrid()

    # read TextGrid
    textgrid.read(textgrid_filename)

    # extract tier names
    tier_names = textgrid.tierNames()


    if textgrid_tier in tier_names: # run over all intervals in the tier
        tier_index = tier_names.index(textgrid_tier)
        textgrid_tier = textgrid[tier_index]
    else: # process first tier
        textgrid_tier = textgrid[0]

    for interval in textgrid_tier:
        if re.search(r'\S', interval.mark()):
            tmp_features_filename = generate_tmp_filename("features")
            tmp_preds = generate_tmp_filename("preds")
            begin = interval.xmin()
            end = interval.xmax()
            features.create_features(wav_filename, tmp_features_filename, begin, end)
            load_estimation_model(tmp_features_filename, tmp_preds, begin, end)
            #easy_call("th load_estimation_model.lua " + tmp_features_filename + ' ' + tmp_preds)
            csv_append_row(tmp_preds, preds_filename)
            delete_temp_files()

    delete_temp_files()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Estimation and tracking of formants.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('formants_file', default='', help="output formant CSV file")
    parser.add_argument('--textgrid_filename', default='', help="get beginning and end times from a TextGrid file")
    parser.add_argument('--textgrid_tier', default='', help="a tier name with portion to process (default first tier)")
    parser.add_argument('--begin', help="beginning time in the WAV file", default=0.0, type=float)
    parser.add_argument('--end', help="end time in the WAV file", default=-1.0, type=float)
    args = parser.parse_args()

    if args.textgrid_filename:
        predict_from_textgrid(args.wav_file, args.formants_file, args.textgrid_filename, args.textgrid_tier)
    else:
        predict_from_times(args.wav_file, args.formants_file, args.begin, args.end)

