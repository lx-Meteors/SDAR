import os, sys, time
WS = "/home/test/yyy/SDAR/agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, WS)

t0 = time.time()
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("import ok %.1fs" % (time.time() - t0))

t0 = time.time()
env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False)
print("env init ok %.1fs" % (time.time() - t0))

# server holds goals
server = env.server
print("num goals:", len(server.goals))
g = server.goals[0]
print("goal keys:", list(g.keys()))
print("instruction:", g.get("instruction_text"))
print("target name (a*):", g.get("name"))
print("target asin:", g.get("asin"), "attributes:", g.get("attributes"))

obs, _ = env.reset(session=0)
print("=== initial obs ===")
print(obs[:600])
aa = env.get_available_actions()
print("available:", aa)

# try a search using the target query
q = server.goals[int(env.session)]["query"] if env.session.isdigit() else g["query"]
print("search query:", g["query"])
state, r, done, info = env.step(f"search[{g['query']}]")
print("=== after search obs ===")
print(state[:800])
print("reward", r, "done", done)
