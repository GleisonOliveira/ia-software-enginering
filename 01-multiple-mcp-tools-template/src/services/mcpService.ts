import { getCSVTOJSONTool } from "../tools/csvToJSONTool.ts";
import { getFSTool } from "../tools/fsTool.ts";
import { getMongoDBTool } from "../tools/mongodbTool.ts";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";

export const getMCPTools = async () => {
  const client = new MultiServerMCPClient({
    mcpServers: {
      ...getMongoDBTool(),
      ...getFSTool(),
    },
    onMessage: (log, source) => {
      console.log(`[${source.server}] ${log.data}`);
    },
  });

  const mcpTools = await client.getTools();

  return [...mcpTools, getCSVTOJSONTool];
};
