window.myNamespace = Object.assign({}, window.myNamespace, {
    tabulator: {
        dataSorted: function (sorter, rows) {
            console.log("Data was sorted");
            let rowData = new Array(rows.length);
            rows.forEach(r => rowData.push(r.getData().index));
            console.log(rowData);
            return rowData;         
        }
    }
});