import React from "react";
import { Header, Image, Styleheet } from "react-native-elements";

import Menu from "./Menu";

const CustomHeader = props => {
  return (
    <Header
       placement="center"
       centerComponent={{ text: props.title, style: { fontWeight: 'bold', fontSize: 26, color: '#fff' } }}
       rightComponent=<Menu navigation={props.navigation} />
       styles={{backgroundColor: 'transparent', flex: 1}}
       statusBarProps = {{ barStyle: "light-content" }}
    />
  );
};

export default CustomHeader;
