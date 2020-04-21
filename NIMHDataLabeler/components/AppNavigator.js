import { createAppContainer } from "react-navigation";
import { createDrawerNavigator } from "react-navigation-drawer";

import Home from "./Home";
import Settings from "./Settings";

const AppNavigator = createDrawerNavigator({
  Home: {
    screen: Home
  },
  Settings: {
    screen: Settings
  },
});

const App = createAppContainer(AppNavigator);

export default App;
