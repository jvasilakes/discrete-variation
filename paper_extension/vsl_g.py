import torch
import config
import train_helper
import data_utils

from models import vsl_g
from model_utils import set_seed
from torch.utils.tensorboard import SummaryWriter

best_dev_res = test_res = 0


def run(e):
    global best_dev_res, test_res

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    dp = data_utils.DataProcessor(experiment=e)
    data, embeddings = dp.process()

    label_logvar_buffer = \
        train_helper.PriorBuffer(data.train[0], e.config.zsize,
                                  experiment=e,
                                  freq=e.config.ufl,
                                  name="label_logvar",
                                  init_path=e.config.prior_file)
    label_mean_buffer = \
        train_helper.PriorBuffer(data.train[0], e.config.zsize,
                                  experiment=e,
                                  freq=e.config.ufl,
                                  name="label_mean",
                                  init_path=e.config.prior_file)

    all_buffer = [label_logvar_buffer, label_mean_buffer]

    e.log.info("labeled buffer size: logvar: {}, mean: {}"
               .format(len(label_logvar_buffer), len(label_mean_buffer)))

    if e.config.use_unlabel:
        unlabel_logvar_buffer = \
            train_helper.PriorBuffer(data.unlabel[0], e.config.zsize,
                                      experiment=e,
                                      freq=e.config.ufu,
                                      name="unlabel_logvar",
                                      init_path=e.config.prior_file)
        unlabel_mean_buffer = \
            train_helper.PriorBuffer(data.unlabel[0], e.config.zsize,
                                      experiment=e,
                                      freq=e.config.ufu,
                                      name="unlabel_mean",
                                      init_path=e.config.prior_file)

        all_buffer += [unlabel_logvar_buffer, unlabel_mean_buffer]

        e.log.info("unlabeled buffer size: logvar: {}, mean: {}"
                   .format(len(unlabel_logvar_buffer),
                           len(unlabel_mean_buffer)))

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if embeddings is None:
        embed_dim = e.config.edim
    else:
        embed_dim = embeddings.shape[1]
    model = vsl_g(
        word_vocab_size=len(data.vocab),
        char_vocab_size=len(data.char_vocab),
        n_tags=len(data.tag_vocab),
        embed_dim=embed_dim,
        embed_init=embeddings,
        experiment=e)

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.summarize:
        writer = SummaryWriter(e.experiment_dir)

    label_batch = data_utils.Minibatcher(
        word_data=data.train[0],
        char_data=data.train[1],
        label=data.train[2],
        batch_size=e.config.batch_size,
        shuffle=False)

    if e.config.use_unlabel:
        unlabel_batch = data_utils.Minibatcher(
            word_data=data.unlabel[0],
            char_data=data.unlabel[1],
            label=data.unlabel[0],
            batch_size=e.config.unlabel_batch_size,
            shuffle=True)

    evaluator = train_helper.Evaluator(data.inv_tag_vocab, model, e)

    e.log.info("Training start ...")
    label_stats = train_helper.Tracker(
        ["loss", "token_loss", "kl_div", "label_loss"])
    unlabel_stats = train_helper.Tracker(
        ["loss", "token_loss", "kl_div"])

    for it in range(e.config.n_iter):
        model.train()
        kl_temp = train_helper.get_kl_temp(e.config.klr, it, 1.0)

        try:
            l_data, l_mask, l_char, l_char_mask, l_label, l_ixs = \
                next(label_batch)
        except StopIteration:
            pass

        lp_logvar = label_logvar_buffer[l_ixs]
        lp_mean = label_mean_buffer[l_ixs]

        l_loss, l_logloss, l_kld, sup_loss, lq_mean, lq_logvar, _ = \
            model(l_data, l_mask, l_char, l_char_mask,
                  l_label, lp_mean, lp_logvar, kl_temp)

        label_logvar_buffer.update_buffer(l_ixs, lq_logvar, l_mask.sum(-1))
        label_mean_buffer.update_buffer(l_ixs, lq_mean, l_mask.sum(-1))

        label_stats.update(
            {"loss": l_loss, "token_loss": l_logloss, "kl_div": l_kld,
             "label_loss": sup_loss}, l_mask.sum())

        if not e.config.use_unlabel:
            model.optimize(l_loss)

        else:
            try:
                u_data, u_mask, u_char, u_char_mask, u_label, u_ixs = \
                    next(unlabel_batch)
            except StopIteration:
                pass

            up_logvar = unlabel_logvar_buffer[u_ixs]
            up_mean = unlabel_mean_buffer[u_ixs]

            u_loss, u_logloss, u_kld, _, uq_mean, uq_logvar, _ = \
                model(u_data, u_mask, u_char, u_char_mask,
                      None, up_mean, up_logvar, kl_temp)

            unlabel_logvar_buffer.update_buffer(
                u_ixs, uq_logvar, u_mask.sum(-1))
            unlabel_mean_buffer.update_buffer(
                u_ixs, uq_mean, u_mask.sum(-1))

            unlabel_stats.update(
                {"loss": u_loss, "token_loss": u_logloss, "kl_div": u_kld},
                u_mask.sum())

            model.optimize(l_loss + e.config.ur * u_loss)

        if (it + 1) % e.config.print_every == 0:
            summary = label_stats.summarize(
                "it: {} (max: {}), kl_temp: {:.2f}, labeled".format(
                    it + 1, len(label_batch), kl_temp))
            if e.config.use_unlabel:
                summary += unlabel_stats.summarize(", unlabeled")
            e.log.info(summary)
            if e.config.summarize:
                writer.add_scalar(
                    "label/kl_temp", kl_temp, it)
                for name, value in label_stats.stats.items():
                    writer.add_scalar(
                        "label/" + name, value, it)
                if e.config.use_unlabel:
                    for name, value in unlabel_stats.stats.items():
                        writer.add_scalar(
                            "unlabel/" + name, value, it)
            label_stats.reset()
            unlabel_stats.reset()
        if (it + 1) % e.config.eval_every == 0:

            e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

            dev_perf, dev_res = evaluator.evaluate(data.dev)

            e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

            if e.config.summarize:
                for n, v in dev_perf.items():
                    writer.add_scalar(
                        "dev/" + n, v, it)

            if best_dev_res < dev_res:
                best_dev_res = dev_res

                e.log.info("*" * 25 + " TEST SET EVALUATION " + "*" * 25)

                test_perf, test_res = evaluator.evaluate(data.test)

                e.log.info("*" * 25 + " TEST SET EVALUATION " + "*" * 25)

                model.save(
                    dev_perf=dev_perf,
                    test_perf=test_perf,
                    iteration=it)

                if e.config.save_prior:
                    for buf in all_buffer:
                        buf.save()

                if e.config.summarize:
                    writer.add_scalar(
                        "dev/best_result", best_dev_res, it)
                    for n, v in test_perf.items():
                        writer.add_scalar(
                            "test/" + n, v, it)
            e.log.info("best dev result: {:.4f}, "
                       "test result: {:.4f}, "
                       .format(best_dev_res, test_res))
            label_stats.reset()
            unlabel_stats.reset()


if __name__ == '__main__':

    args = config.get_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()

    def exit_handler(*args):
        print(args)
        print(f"best dev result: {best_dev_res:.4f}, "
              f"test result: {test_res:.4f}")
        exit()

    train_helper.register_exit_handler(exit_handler)
    set_seed(args.random_seed)
    with train_helper.Experiment(args, args.prefix) as e:
        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        run(e)
