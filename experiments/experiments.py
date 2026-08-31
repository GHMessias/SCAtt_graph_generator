import sys
sys.path.append("../")

from experiments.func import *   # arguments(), train_gae_kmeans_nmi, etc.
from models.SCatt import SCAttGenerator
from models.GenCAT import GenCATGenerator
from models.SkyMap import SkyMapGenerator
from models.chunglu import ChungLuGenerator

from core.attributed_graph import AttributedGraph

from analysis.sfanalysis import PowerLawAnalysis

import networkit as nk

from torch_geometric.data import Data
from torch_geometric.utils import to_networkit

import subprocess

# ----------------------------
# Core
# ----------------------------
def main():
    args = arguments()

    if args.experiment == "topology_rho":

        # if not 1.0 % args.grid_step == 0:
        #     raise ValueError("grid step deve andar de 0 até 1.0")
        
        # 1) cria a pasta do experimento
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        exp_dir = Path(f"experiments/results/topologic/topology_rho_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 2) salva summary do modelo base (sem rho)
        save_experiment_summary(args, exp_dir)

        df = pd.DataFrame()

        tmp_value = int(round(1.0 / args.grid_step))

        alpha_range = [i * args.grid_step for i in range(tmp_value + 1)]
        lambda_range = [i * args.grid_step for i in range(tmp_value + 1)]

        for rho_value in args.rho_list:
            for exc in range(args.num_executions):
                for aa in alpha_range:
                    for ll in lambda_range:

                        # 3) gera o grafo com rho atual
                        G = SCAttGenerator(seed = 2026).generate(
                            n=args.n,
                            e=torch.tensor(args.num_edges),
                            y=args.y,
                            k=args.k,
                            A_in=[torch.tensor(x) for x in args.A_in],
                            dst=args.dst,
                            rho=rho_value,
                            alpha_topo=aa,
                            alpha_feat=ll,
                            A_out=torch.tensor(args.A_out),
                            S=[torch.tensor(x) for x in args.S],
                            d=args.d
                        )

                        dt = G.to_data_pytorch()

                        # 4) roda o treino + avaliação
                        # IDEAL: train_gae_kmeans_nmi retornar também o modelo
                        out = train_gae_kmeans_nmi(data=dt, hidden_channels=32, embedding_dim=16)

                        # 5) salva o grafo Data
                        save_graph_data(dt, exp_dir, exc, rho_value)

                        # 6) salva o modelo (se o out retornar o modelo)
                        # Esperado: out["model"] = model treinado
                        if isinstance(out, dict) and "model" in out and out["model"] is not None:
                            save_model(out["model"], exp_dir, exc, rho_value)
                        else:
                            # (a gente vai ajustar depois no train_gae_kmeans_nmi)
                            pass

                        # 7) adiciona resultados ao dataframe
                        df = append_result(df, out, dt, exc, rho_value, ll, aa)

                        print(f"[exc={exc}] rho={rho_value} nmi={out['nmi']}, alpha_feat={ll}, alpha_topo={aa}")

                        # 9) salva métricas no final
                        save_metrics(df, exp_dir)

    if args.experiment == "sfanalysis":

        # Load base graph
        data_to_mimic = torch.load(args.base_graph_path)[0]
        data_to_mimic = Data(
            x=data_to_mimic["x"],
            y=data_to_mimic["y"],
            edge_index=data_to_mimic["edge_index"]
        )
        data_to_mimic = AttributedGraph(
            x=data_to_mimic.x,
            y=data_to_mimic.y,
            graph=to_networkit(data_to_mimic.edge_index, directed=False)
        )
        print(data_to_mimic)

        df = pd.DataFrame()

        models = {
            "GenCAT": GenCATGenerator(),
            "SkyMap": SkyMapGenerator(),
            "SCAtt": SCAttGenerator(),
            "chung-lu": ChungLuGenerator()
        }

        # ✅ Cria 1 pasta por execução (evita sobrescrita e “sumir” resultados)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")  # microssegundos
        exp_dir = Path(f"experiments/results/sfanalysis/sfanalysis_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Salva summary do experimento 1x (não depende do modelo)
        save_experiment_summary(args, exp_dir)

        for clone_model in args.clone_models:
            print(f"STARTING CLONE GRAPH FOR MODEL {clone_model}")

            # Set the number of nodes to clone
            if args.num_aug_nodes is not None:
                # print(type(data_to_mimic.graph.numberOfNodes()))
                num_aug_nodes = int(args.num_aug_nodes * data_to_mimic.graph.numberOfNodes())
            else:
                num_aug_nodes = None

            graph = models[clone_model].mimic(data_to_mimic, num_nodes = num_aug_nodes)

            # ✅ Salva o grafo gerado (1 por modelo, com nome diferente)
            save_graph_data(graph.to_data_pytorch(), exp_dir, model_name=clone_model)

            # Métricas dos subgrafos do grafo gerado
            for index, gg in graph.subgraphs.items():

                deg = nk.centrality.DegreeCentrality(gg)
                deg.run()

                degrees = np.array(deg.scores())
                degrees = degrees[degrees > 0]

                pla = PowerLawAnalysis(degrees)
                pla.evaluate_distribution()

                results = pla.summary()
                results["subgraph"] = index
                results["model"] = clone_model

                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

                # (opcional) salvar a cada iteração; ou melhor salvar no final
                save_metrics(df, exp_dir)

        # ✅ Avaliar o original (comparação) no MESMO diretório do experimento
        for index, gg in data_to_mimic.subgraphs.items():
            deg = nk.centrality.DegreeCentrality(gg)
            deg.run()

            degrees = np.array(deg.scores())
            degrees = degrees[degrees > 0]

            pla = PowerLawAnalysis(degrees)
            pla.evaluate_distribution()

            results = pla.summary()
            results["subgraph"] = index
            results["model"] = args.base_graph_path

            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

            save_metrics(df, exp_dir)

        return


    if args.experiment == "DGCluster":            
        def euristic_distribution(x):
            if 1 < x < 499:
                return "uniform"
            if 499 < x < 999:
                return "normal"
            else:
                return "power_law"
            
        def euristic_subcomm_interaction(x):
            if 1 < x < 499:
                return torch.tensor([1])
            if 500 < x < 999:
                return torch.tensor([[0.8,0.2], [0.2,0.8]])
            else:
                return torch.tensor([[0.99,0.01,0.01], [0.01, 0.99, 0.01], [0.01,0.01,0.99]])
            
        def euristic_subcomm_prob(x):
            if 1 < x < 499:
                return torch.tensor([1])
            if 500 < x < 999:
                return torch.tensor([0.75, 0.25])
            else:
                return torch.tensor([0.33, 0.33, 0.33])

        if args.dgcluster_test == "test1":
            for num_com in range(2, args.max_num_communities + 1):
                path = "DGCluster_SCAtt_datasets/SCAtt_numcom_{len(y)}"
                y = torch.tensor([int(args.default_node_values / sum(list(range(1,num_com+1)))) * index for index in range(1,num_com+1)][::-1])
                e = torch.tensor([val*4 if val >= 1000 else val*3 if 500 < val < 1000 else int(val*2.5) for val in y])
                k= len(y)
                dst = [euristic_distribution(x) for x in y]
                A_in = [euristic_subcomm_interaction(x) for x in y]
                S = [euristic_subcomm_prob(x) for x in y]

                A_out = torch.ones(size = (len(y), len(y))) - torch.eye(n = len(y))

                path = f"experiments/DGCluster_SCAtt_datasets/SCAtt_numcom_{len(y)}.pt"

                if os.path.exists(path):
                    print(f"Skipping, graph alredy exists in {path}")
                else:
                    graph = SCAttGenerator().generate(n=y.sum().item(), y=y, k=k, e=e, A_in=A_in, dst=dst, rho=int(e.sum().item() * 1.2), A_out=A_out, S=S, alpha_powerlaw=1.3, d=256, alpha_topo=0.5, alpha_feat=0.3)
                    dt = graph.to_data_pytorch()
                    torch.save(dt, path)

                for lbd in [0.1, 0.5, 0.8]:
                    # Classificar com o DGCluster
                    subprocess.run([
                            "python3", "DGCluster/main.py",
                            "--dataset", "custom",
                            "--custom_dataset_path", str(path),
                            "--lam", str(lbd),
                            "--epochs", "100",
                            "--SCAtt_lbd", str(0.001),
                            "--SCAtt_alpha", str(0.5),
                            "--SCAtt_rho", str(0.3),
                            # "--seed", str(seed)
                        ], check=True)
                    

if __name__ == "__main__":
    main()
