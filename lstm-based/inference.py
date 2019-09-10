from utils import *
from model import *
from solver import *

SEED = 2019
path = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
output_path = './'
EMBEDDING_FILES = [
    # '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    # '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=220)
parser.add_argument('--vocab-size', type=int, default=100000)
parser.add_argument('--nb-models', type=int, default=5,
                    help='number of models (folds) to ensemble')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--enable-ckpt-ensemble', type=bool, default=0)
parser.add_argument('--ckpt-per-fold', type=bool, default=1)
args = parser.parse_args()


def convert_dataframe_to_bool(df):
    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    bool_df = df.copy()
    for col in [label_column] + identity_columns + aux_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def load_and_preproc():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')
    train_df[identity_columns] = train_df[identity_columns].copy().fillna(0)

    print('cleaning text...')
    t0 = time.time()
    train_df[text_column] = train_df[text_column].apply(clean_text)
    test_df[text_column] = test_df[text_column].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    return train_df[text_column], test_df[text_column], test_df['id']



## main()
np.random.seed(SEED)
train_seq, x_test, test_id = load_and_preproc()

print('tokenizing...')
t0 = time.time()
word_to_idx, idx_to_word = word_idx_map(train_seq.tolist()+x_test.tolist(), args.vocab_size)
x_test = tokenize(x_test, word_to_idx, args.maxlen)
print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))


print('loading embeddings...')
t0 = time.time()
embed_mat = np.concatenate(
    [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=-1)
# embed_mat = np.mean(
#     [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=0)
print('loading complete in {:.0f} seconds.'.format(time.time()-t0))


# preparation
test_preds = []     # for the predictions on the testset
test_loader, test_original_indices = prepare_loader(x_test, split='test')

# model setup
torch.cuda.empty_cache()
models = torch.load(model_path+'models.pt')

# inference
if args.enable_ckpt_ensemble:
    ckpt_weights = [2**e for e in range(args.epochs)]
    for i in range(args.nb_models):
        single_test_preds = []
        for e in range(args.epochs):
            model = JigsawNet(*embed_mat.shape, 128, embed_mat)
            model = model.to(device)
            model.load_state_dict(models[f'fold_{i}_epk_{e}'])
            test_scores = eval_model(model, test_loader)
            single_test_preds.append(test_scores[test_original_indices])
        test_preds.append(np.average(single_test_preds, weights=ckpt_weights, axis=0))

if args.ckpt_per_fold:
    for i in range(args.nb_models):
        model = JigsawNet(*embed_mat.shape, 128, embed_mat)
        model = model.to(device)
        model.load_state_dict(models[f'fold_{i}'])
        test_scores = eval_model(model, test_loader)
        test_preds.append(test_scores[test_original_indices])


# submission file
test_preds = np.mean(test_preds, 0)
submission = pd.DataFrame.from_dict({
    'id': test_id,
    'prediction': test_preds
})
submission.to_csv('submission.csv', index=False)
